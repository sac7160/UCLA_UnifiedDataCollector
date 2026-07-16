import android.annotation.SuppressLint;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.audiofx.AcousticEchoCanceler;
import android.media.audiofx.AutomaticGainControl;
import android.media.audiofx.NoiseSuppressor;
import android.util.Log;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class DataRecorder {

    public static final int  AUDIO_SAMPLE_RATE = Utilities.SamplingRate;
    public static final int  MAX_FRAME_SIZE    = Utilities.SamplingRate / 25;
    private static final String TAG            = "Data Recorder";

    // IMU buffer: flush every 20 samples (~200ms at 100Hz)
    private static final int IMU_CHUNK_SIZE = 20;
    private final List<String> accBuffer = new ArrayList<>();
    private final List<String> gyrBuffer = new ArrayList<>();

    // IP saved when audio streaming starts — reused for IMU streaming
    private String currentIp = "";

    public static DataRecorder dataRecorder = new DataRecorder();

    static int device                       = MediaRecorder.AudioSource.UNPROCESSED;
    private static final int CHANNEL        = AudioFormat.CHANNEL_IN_MONO;
    private static final int FORMAT         = AudioFormat.ENCODING_PCM_16BIT;
    private static final int RECORDING_RATE = Utilities.SamplingRate;
    private AudioRecord recorder;

    long recordingStartTime;
    long recordingStopTime;

    private int BUFFER_SIZE = AudioRecord.getMinBufferSize(RECORDING_RATE, CHANNEL, FORMAT);

    boolean currentlyRecordingAudio = false;
    boolean sendMsg;

    // Number of bytes used for the per-frame capture timestamp prefix
    // (a big-endian long / System.currentTimeMillis()).
    private static final int FRAME_TS_BYTES = 8;

    // ── Persistent connection ────────────────────────────────────────────────
    //
    // Previously, every single message (each ~40ms audio frame, every IMU
    // chunk) opened a BRAND NEW Socket via AsyncTask.execute(), which by
    // default runs on a single shared SERIAL executor. Two problems fell out
    // of that, confirmed by the captured data (a ~30s recording taking
    // ~39-48s of real time to fully arrive, with content intact but badly
    // time-smeared):
    //   1. Socket connect() had no explicit timeout, so a single slow/failed
    //      connection attempt could block for a long OS-default timeout.
    //   2. Because the executor is serial, that one stuck task blocked EVERY
    //      later-queued message (audio and IMU alike) behind it, and capture
    //      kept piling up behind the block — a small stall snowballs into a
    //      much larger one.
    //
    // This does NOT make the pipeline immune to network delay — a genuinely
    // dead WiFi link for several seconds still means that interval's data
    // can't be delivered, and a very long, sustained outage will eventually
    // overflow the bounded send queue and start dropping the oldest pending
    // messages (see MAX_QUEUE_SIZE below). What it does fix is the snowball
    // effect: a transient stall no longer blocks everything queued behind
    // it, and reconnects happen quickly (CONNECT_TIMEOUT_MS) instead of
    // waiting on an OS-default timeout that can run into tens of seconds.
    private static final int WATCH_PORT           = 50005;
    private static final int CONNECT_TIMEOUT_MS   = 1000;
    private static final int RECONNECT_BACKOFF_MS = 200;
    private static final int MAX_QUEUE_SIZE        = 500;

    private final BlockingQueue<byte[]> sendQueue = new LinkedBlockingQueue<>();
    private Thread writerThread;
    private volatile boolean writerRunning = false;

    private Socket persistentSocket;
    private DataOutputStream persistentOut;

    private synchronized void startPersistentConnectionIfNeeded(String ip) {
        currentIp = ip;
        if (writerRunning) return;
        writerRunning = true;
        writerThread = new Thread(this::writerLoop, "WatchStreamWriter");
        writerThread.setDaemon(true);
        writerThread.start();
    }

    /**
     * Blocks (up to timeoutMs) until the persistent connection is actually
     * established. Used before starting audio capture so that RTBGN's
     * timestamp — captured right as frame 0 is read — doesn't get sent out
     * only after an initial TCP handshake delay of up to CONNECT_TIMEOUT_MS.
     * Without this, rtbgn_watch_ms (captured pre-connection) and
     * rtbgn_pc_sec (captured on arrival, post-connection) end up biased by
     * however long the first connect() took — which showed up as a fixed
     * ~1s (== CONNECT_TIMEOUT_MS) offset across an entire session's worth of
     * watch_ts_ms → PC-time alignment.
     */
    private boolean waitForConnection(long timeoutMs) {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (persistentSocket != null && persistentSocket.isConnected() && !persistentSocket.isClosed()) {
                return true;
            }
            try {
                Thread.sleep(10);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        Log.w(TAG, "waitForConnection timed out after " + timeoutMs + "ms — proceeding anyway "
                + "(RTBGN alignment may be off for this session)");
        return false;
    }

    /** Call when the whole recording session is done, to close the
     * connection after any remaining queued messages have drained. */
    public void closeConnection() {
        writerRunning = false;
        if (writerThread != null) {
            try {
                writerThread.join(1000);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void enqueue(byte[] msg) {
        if (sendQueue.size() >= MAX_QUEUE_SIZE) {
            sendQueue.poll();   // drop the oldest pending message — better to
                                 // stay current than build an ever-growing
                                 // backlog during a sustained outage
        }
        sendQueue.offer(msg);
    }

    private boolean ensureConnected() {
        if (persistentSocket != null && persistentSocket.isConnected() && !persistentSocket.isClosed()) {
            return true;
        }
        try {
            Socket s = new Socket();
            s.setTcpNoDelay(true);   // disable Nagle's algorithm — these are small, latency-sensitive messages
            s.connect(new InetSocketAddress(currentIp, WATCH_PORT), CONNECT_TIMEOUT_MS);
            persistentOut     = new DataOutputStream(new BufferedOutputStream(s.getOutputStream()));
            persistentSocket  = s;
            Log.w(TAG, "connected to " + currentIp + ":" + WATCH_PORT);
            return true;
        } catch (IOException e) {
            closeSocketQuietly();
            try {
                Thread.sleep(RECONNECT_BACKOFF_MS);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
            return false;
        }
    }

    private void closeSocketQuietly() {
        if (persistentSocket != null) {
            try {
                persistentSocket.close();
            } catch (IOException ignored) {
            }
        }
        persistentSocket = null;
        persistentOut = null;
    }

    private void writerLoop() {
        while (writerRunning || !sendQueue.isEmpty()) {
            // Try to connect eagerly, independent of whether a message is
            // waiting — this is what lets waitForConnection() (called before
            // recording starts) actually observe a connected socket, instead
            // of the connection only being attempted once the first message
            // happens to be enqueued.
            if (!ensureConnected()) {
                continue;   // ensureConnected() already backed off briefly on failure
            }

            byte[] msg;
            try {
                msg = sendQueue.poll(200, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                continue;
            }
            if (msg == null) continue;

            try {
                // 4-byte big-endian length prefix, then the raw message —
                // lets the PC side split messages apart on a long-lived
                // connection instead of relying on "one message per TCP
                // connection" (which is what forced the old per-message
                // socket design in the first place).
                persistentOut.writeInt(msg.length);
                persistentOut.write(msg);
                persistentOut.flush();
            } catch (IOException e) {
                Log.w(TAG, "write failed, will reconnect: " + e);
                closeSocketQuietly();
            }
        }
        closeSocketQuietly();
    }

    // ── Audio streaming ──────────────────────────────────────────────────────

    public void startStreamingAudio(String ip, int duration, int id) {
        currentIp = ip;   // save IP for IMU streaming
        startPersistentConnectionIfNeeded(ip);
        currentlyRecordingAudio = true;
        sendMsg = true;
        startStreaming(ip, duration, id);
    }

    public void stopStreamingAudio() {
        stopStreamingAudio(true);
    }

    private void stopStreamingAudio(boolean sendMsgIn) {
        sendMsg = sendMsgIn;
        currentlyRecordingAudio = false;
        // Flush remaining IMU samples on stop
        flushImu("IMUAC", accBuffer);
        flushImu("IMUGY", gyrBuffer);
    }

    private void startStreaming(final String ip, final int duration, final int id) {
        recordingStartTime = recordingStopTime = -1;
        Thread streamThread = new Thread(new Runnable() {
            @SuppressLint("MissingPermission")
            @Override
            public void run() {
                BUFFER_SIZE = MAX_FRAME_SIZE * 2;
                int maxPackets = duration * (AUDIO_SAMPLE_RATE / MAX_FRAME_SIZE);

                int extraBytes = 10;
                byte[] buffer = new byte[extraBytes + (BUFFER_SIZE * maxPackets)];

                buffer[0]='S'; buffer[1]='O'; buffer[2]='U'; buffer[3]='N'; buffer[4]='D';
                byte[] num = Utilities.leftPad(Integer.toString(id), 5).getBytes();
                buffer[5]=num[0]; buffer[6]=num[1]; buffer[7]=num[2];
                buffer[8]=num[3]; buffer[9]=num[4];

                int count = 0;

                try {
                    recorder = new AudioRecord(device, RECORDING_RATE, CHANNEL, FORMAT, BUFFER_SIZE);
                    int sessionId = recorder.getAudioSessionId();

                    if (NoiseSuppressor.isAvailable())
                        NoiseSuppressor.create(sessionId).setEnabled(false);
                    if (AutomaticGainControl.isAvailable())
                        AutomaticGainControl.create(sessionId).setEnabled(false);
                    if (AcousticEchoCanceler.isAvailable())
                        AcousticEchoCanceler.create(sessionId).setEnabled(false);

                    // Wait here (on this background thread, not the caller's)
                    // for the persistent connection to actually be up before
                    // recording starts. Otherwise RTBGN's timestamp — taken
                    // right as frame 0 is captured — gets sent only after an
                    // initial TCP handshake delay of up to CONNECT_TIMEOUT_MS,
                    // biasing the entire session's watch_ts_ms → PC-time
                    // alignment by however long that handshake took.
                    waitForConnection(2000);

                    recorder.startRecording();
                    recordingStartTime = System.currentTimeMillis();
                    Log.w(TAG, "Log/ maxPackets: " + maxPackets);

                    while (currentlyRecordingAudio) {
                        int start = extraBytes + (BUFFER_SIZE * count);
                        int read = recorder.read(buffer, start, BUFFER_SIZE);
                        // Plain real-time measurement, with NO fixed
                        // correction applied. Two different correction
                        // attempts were tried here (a pure count*duration
                        // calculation, and subtracting one frameDurationMs
                        // from this reading) and both empirically made
                        // alignment worse than this plain version — meaning
                        // the true relationship between this reading and the
                        // frame's actual capture time isn't a simple,
                        // guessable constant. Fix this only based on an
                        // actual measurement (e.g. logging both a
                        // pre-read() and post-read() timestamp across many
                        // frames to see how read() latency really behaves
                        // on this device), not another assumption.
                        long frameTs = System.currentTimeMillis();

                        if (count == maxPackets) {
                            if (Utilities.IsRealtimeStreaming) {
                                buffer[0]='R'; buffer[1]='T'; buffer[2]='E'; buffer[3]='N'; buffer[4]='D';
                                // Unlike frame timestamps above, this reads the
                                // real wall clock on purpose: RTEND's whole
                                // job is to measure how much real time has
                                // actually elapsed, so the PC side can compare
                                // it against RTBGN and correct for any drift
                                // between the nominal audio-clock timeline and
                                // real time (see _recalibrate_session_trials
                                // on the PC side).
                                long t = System.currentTimeMillis();    //for sync with surface microphone
                                buffer[5]=(byte)(t>>56); buffer[6]=(byte)(t>>48);
                                buffer[7]=(byte)(t>>40); buffer[8]=(byte)(t>>32);
                                buffer[9]=(byte)(t>>24); buffer[10]=(byte)(t>>16);
                                buffer[11]=(byte)(t>>8); buffer[12]=(byte)(t);
                                send_request(ip, buffer, 13);
                            }
                            currentlyRecordingAudio = false;
                        } else {
                            if (Utilities.IsRealtimeStreaming) {
                                int end = start + BUFFER_SIZE;

                                // Every audio frame — including frame 0 — is sent
                                // as [8-byte big-endian timestamp][BUFFER_SIZE
                                // bytes of PCM audio]. Frame 0's audio used to be
                                // silently dropped (only the RTBGN header was
                                // sent for it); it's now sent like every other
                                // frame, just with its own timestamp attached.
                                byte[] framePkt = new byte[FRAME_TS_BYTES + BUFFER_SIZE];
                                framePkt[0] = (byte)(frameTs >> 56);
                                framePkt[1] = (byte)(frameTs >> 48);
                                framePkt[2] = (byte)(frameTs >> 40);
                                framePkt[3] = (byte)(frameTs >> 32);
                                framePkt[4] = (byte)(frameTs >> 24);
                                framePkt[5] = (byte)(frameTs >> 16);
                                framePkt[6] = (byte)(frameTs >> 8);
                                framePkt[7] = (byte)(frameTs);
                                System.arraycopy(buffer, start, framePkt, FRAME_TS_BYTES, BUFFER_SIZE);
                                send_request(ip, framePkt, framePkt.length);

                                if (count == 0) {
                                    // Announce stream start via RTBGN, as before.
                                    // This overwrites buffer[0..12], which overlaps
                                    // frame 0's storage slot (buffer[10..12]) — that's
                                    // harmless now since frame 0's audio was already
                                    // copied out into framePkt above before this point.
                                    //
                                    // Reuses this iteration's frameTs (not a
                                    // fresh System.currentTimeMillis() call)
                                    // so RTBGN's watch_ms is exactly identical
                                    // to frame 0's own timestamp — both are
                                    // the same corrected real-time reading.
                                    buffer[0]='R'; buffer[1]='T'; buffer[2]='B'; buffer[3]='G'; buffer[4]='N';
                                    long t = frameTs;
                                    buffer[5]=(byte)(t>>56); buffer[6]=(byte)(t>>48);
                                    buffer[7]=(byte)(t>>40); buffer[8]=(byte)(t>>32);
                                    buffer[9]=(byte)(t>>24); buffer[10]=(byte)(t>>16);
                                    buffer[11]=(byte)(t>>8); buffer[12]=(byte)(t);
                                    send_request(ip, buffer, 13);
                                }
                            }
                            count++;
                        }
                    }
                    recordingStopTime = System.currentTimeMillis();
                } catch (Exception e) {
                    Log.w(TAG, "TCP Streamer Exception: " + e);
                }
                recorder.stop();
                if (sendMsg && !Utilities.IsRealtimeStreaming)
                    send_request(ip, buffer, extraBytes + (BUFFER_SIZE * count));
                recorder.release();
            }
        });
        streamThread.start();
    }

    // ── IMU real-time streaming ───────────────────────────────────────────────

    /**
     * Buffer accelerometer sample; flush every IMU_CHUNK_SIZE samples.
     * Uses currentIp saved from startStreamingAudio().
     */
    public synchronized void addAccRealtime(String sample) {
        accBuffer.add(sample);
        if (accBuffer.size() >= IMU_CHUNK_SIZE) {
            flushImu("IMUAC", accBuffer);
        }
    }

    /**
     * Buffer gyroscope sample; flush every IMU_CHUNK_SIZE samples.
     */
    public synchronized void addGyrRealtime(String sample) {
        gyrBuffer.add(sample);
        if (gyrBuffer.size() >= IMU_CHUNK_SIZE) {
            flushImu("IMUGY", gyrBuffer);
        }
    }

    /**
     * Packet format: "IMUAC" or "IMUGY" (5-char header)
     *              + samples joined by "|"
     * Each sample : "x y z timestamp"
     */
    private void flushImu(String header, List<String> buffer) {
        if (buffer.isEmpty() || currentIp.isEmpty()) return;
        StringBuilder sb = new StringBuilder(header);
        for (int i = 0; i < buffer.size(); i++) {
            if (i > 0) sb.append("|");
            sb.append(buffer.get(i));
        }
        buffer.clear();
        sendMsgString(currentIp, sb.toString());
    }

    // ── TCP helpers ───────────────────────────────────────────────────────────
    //
    // Both of these now enqueue onto the single persistent connection and
    // return immediately, instead of opening a new Socket per call and
    // blocking the caller's thread on network I/O. Signatures are unchanged
    // so existing call sites elsewhere in the app keep working as-is.

    public void send_request(String ip, byte[] buf, int bufSize) {
        startPersistentConnectionIfNeeded(ip);
        enqueue(Arrays.copyOf(buf, bufSize));
    }

    public static void sendMsgString(String ip, String s) {
        dataRecorder.startPersistentConnectionIfNeeded(ip);
        dataRecorder.enqueue(s.getBytes());
    }
}