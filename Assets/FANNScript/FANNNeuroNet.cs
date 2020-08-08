using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using FANN_Library;

public class FANNNeuroNet : MonoBehaviour {

    public FANNClass FANN;
    public float[] input;
    public float[] output;
    
    // Use this for initialization
    void Start () {
        FANN = new FANNClass(true, 3, 3, 10, 1);
	}

    public void TrainNN(float[] input, float[] output)
    {
        float MSE =  FANN.TrainOnIO(input, output, 100, true, true);
        Debug.Log(FANN.LogResult);
        //return MSE;
    }
    public void TrainNN()
    {
        float MSE = FANN.TrainOnIO(input, output, 100, true, true);
        Debug.Log(FANN.LogResult);
        //return MSE;
    }
    // Update is called once per frame
    void Update () {
		
	}
}
