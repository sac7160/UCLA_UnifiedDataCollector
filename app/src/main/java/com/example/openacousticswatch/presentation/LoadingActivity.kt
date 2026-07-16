// (c) 2025, KAIST, WIT_LAB, Jiwan Kim (jiwankim@kaist.ac.kr, kjwan4435@gmail.com)

package com.example.openacousticswatch.presentation

import DataRecorder.dataRecorder
import Utilities.RecordingTime
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import androidx.activity.ComponentActivity
import com.example.openacousticswatch.R

class LoadingActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_loading)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        dataRecorder.startStreamingAudio(Utilities.IP, RecordingTime, Utilities.TrialEndCounter)

        Handler(Looper.getMainLooper()).postDelayed({
            val capture = Intent(this@LoadingActivity, CaptureActivity::class.java)
            startActivity(capture)
            finish()
        }, 1000)
    }
}