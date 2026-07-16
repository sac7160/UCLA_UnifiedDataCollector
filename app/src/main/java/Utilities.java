// (c) 2025, KAIST, WIT_LAB, Jiwan Kim (jiwankim@kaist.ac.kr, kjwan4435@gmail.com)

import android.os.Build;

import androidx.annotation.RequiresApi;

public class Utilities {
    static public String IP = "192.168.0.110";
    static public String SUB_ID = "0";

    static public int RecordingTime = 30;
    static public boolean IsRecordingIMU = true;
    static public boolean IsRealtimeStreaming = true;
    static public int SamplingRate = 48000;
    static public int NofBlocksReps = 30;
    static public int NofBlocks = NofBlocksReps;
    static public int BlockCounter = 0;
    static public int TrialCounter = 0;
    static public int TrialEndCounter = 0;

    static public int TargetGroundTruth = 1;
    static public int TargetReps = 1;
    static public int NofTrials = TargetGroundTruth*TargetReps;
    static public int[] TargetGroundTruths = {0};


    @RequiresApi(api = Build.VERSION_CODES.O)
    static public String getDateTS()
    {
        String spacer = "";
        String ret = java.time.LocalDate.now().toString();
        ret = ret.substring(ret.indexOf("-")+1);
        ret = ret.replace("-", spacer);

        String ret2 = java.time.LocalTime.now().toString();
        ret2 = ret2.replace(":", spacer);
        ret2 = ret2.substring(0, ret2.indexOf("."));

        return ret+"_"+ret2;
    }

    static public String leftPad(String result, int padNum)
    {
        StringBuilder sb = new StringBuilder();
        int rest = padNum - result.length();
        for (int i = 0; i < rest; i++)
        {
            sb.append("0");
        }
        sb.append(result);
        return sb.toString();
    }
}
