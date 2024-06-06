package com.example.NMT;

import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

public class Decoder {
    private static final String TAG = Decoder.class.getName();
    public static String getText(ArrayList<Long> result_idx, JSONObject idx2word)
    {
        StringBuilder outputs = new StringBuilder();
        try {
            for(int i = 0;i < result_idx.size(); i++)
            {
                String targetWord = idx2word.getString(""+result_idx.get(i));
                if(targetWord.startsWith("▁"))
                {
                    targetWord = targetWord.replace("▁"," ");
                }
                outputs.append(targetWord);
            }
        }
        catch (JSONException e) {
            Log.e(TAG, "JSONException ", e);
        }
        return outputs.toString();
    }
}
