// (c) 2025, KAIST, WIT_LAB, Jiwan Kim (jiwankim@kaist.ac.kr, kjwan4435@gmail.com)

package com.example.openacousticswatch.presentation

import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import android.widget.Button
import androidx.activity.ComponentActivity
import com.example.openacousticswatch.R


class TapActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_tap)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val tap: Button = findViewById(R.id.sub_button)
        tap.setOnClickListener{
            val loading = Intent(this, LoadingActivity::class.java)
            startActivity(loading)
            finish()
        }
    }
}