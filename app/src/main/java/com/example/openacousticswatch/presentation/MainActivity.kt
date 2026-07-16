// (c) 2025, KAIST, WIT_LAB, Jiwan Kim (jiwankim@kaist.ac.kr, kjwan4435@gmail.com)

package com.example.openacousticswatch.presentation

import DataRecorder.sendMsgString
import Utilities.BlockCounter
import Utilities.TrialCounter
import Utilities.TrialEndCounter
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import androidx.activity.ComponentActivity
import com.example.openacousticswatch.R

class MainActivity : ComponentActivity() {

    private val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // View Allocation
        var textIP: EditText = findViewById<EditText>(R.id.et_ip).also {
            it.setText(Utilities.IP)
        }
        var textSubnum: EditText = findViewById<EditText>(R.id.et_subnum)
        val startBtn: Button = findViewById(R.id.btn_start)

        BlockCounter = 0;
        TrialCounter = 0;
        TrialEndCounter = 0;

        startBtn.setOnClickListener {
            Utilities.IP = textIP.getText().toString()
            Utilities.SUB_ID = textSubnum.getText().toString()

            val MSG = "SUBID" + Utilities.leftPad(Utilities.SUB_ID, 5) + Utilities.getDateTS()
            sendMsgString(Utilities.IP, MSG)

            Log.w(TAG, "Log/ IP: " + Utilities.IP + ", MSG: " + MSG)

            val intent = Intent(this@MainActivity, BlockActivity::class.java)
            startActivity(intent)

            finish()
        }
    }
}