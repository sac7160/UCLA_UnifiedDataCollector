// (c) 2025, KAIST, WIT_LAB, Jiwan Kim (jiwankim@kaist.ac.kr, kjwan4435@gmail.com)
// Modified: IMU real-time streaming added; recording now runs until manually
// stopped instead of a fixed duration.

package com.example.openacousticswatch.presentation

import DataRecorder.dataRecorder
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.view.WindowManager
import android.widget.Button
import androidx.activity.ComponentActivity
import com.example.openacousticswatch.R
import com.example.openacousticswatch.presentation.BlockActivity.Companion.trialSet

class CaptureActivity : ComponentActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_capture)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setUpSensorStuff()

        findViewById<Button>(R.id.btn_stop).setOnClickListener { endActivity() }

        trialSet.startTrial(System.currentTimeMillis())
    }

    fun endActivity() {
        setOffSensorStuff()
        trialSet.endTrial(System.currentTimeMillis())
        dataRecorder.stopStreamingAudio()
        val saving = Intent(this@CaptureActivity, SavingActivity::class.java)
        startActivity(saving)
        finish()
    }

    private fun setUpSensorStuff() {
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.also {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.also {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
        sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)?.also {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
    }

    private fun setOffSensorStuff() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        val timestamp = System.currentTimeMillis()
        when (event?.sensor?.type) {

            Sensor.TYPE_ACCELEROMETER -> {
                val x = event.values[0]
                val y = event.values[1]
                val z = event.values[2]
                val sample = "$x $y $z $timestamp"
                trialSet.addAcc(sample)
                dataRecorder.addAccRealtime(sample)
            }

            Sensor.TYPE_GYROSCOPE -> {
                val x = event.values[0]
                val y = event.values[1]
                val z = event.values[2]
                val sample = "$x $y $z $timestamp"
                trialSet.addGyr(sample)
                dataRecorder.addGyrRealtime(sample)
            }

            Sensor.TYPE_MAGNETIC_FIELD -> {
                val x = event.values[0]
                val y = event.values[1]
                val z = event.values[2]
                trialSet.addMag("$x $y $z $timestamp")
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    override fun onDestroy() {
        sensorManager.unregisterListener(this)
        super.onDestroy()
    }
}
