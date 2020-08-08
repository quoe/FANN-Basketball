using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using FANN_Library;
using System.IO;

public class FANNNeuroNet : MonoBehaviour {
    public FANNClass FANN;
    public float[] input;
    public float[] output;
    public bool Scale = false;
    public bool StringLayers = false;
    public bool RandomFANNParam = false;
    public string Layers = "3, 10, 10, 10, 10, 1";
    public bool isCollectData = false;
    public bool LoadTrainData = false;
    public bool LoadBrainData = false;
    public bool isUseBrainFunc = false;
    public int BrainMaxIterations = 1000;
    public double[] BrainRndMin;
    public double[] BrainRndMax;
    public double BrainError = 0.1;
    public bool ShowBrainLog = true;
    public int TrainCount = 1000;
    public int TrainEachNum = 100;
    public uint FANNLayers = 3;
    public uint FANNHiddenNeurons = 10;
    private int TrainListLength;
    private float MSE;
    public string LogMessage;
    public string ParentName;
    public string ResultInfo;
    private string TrainFileName = "_TrainFile.train";
    private string NetFileName = "_NetFile.net";
    //private IEnumerator coroutine;

    // Use this for initialization
    void Start () {
        if (!LoadBrainData)
        {
            if (RandomFANNParam)
            {
                BrainMaxIterations = Random.Range(1, 10000);
                BrainError = Random.Range(0.0f, 1.0f);
                TrainCount = Random.Range(10, 10000);
                TrainEachNum = Random.Range(10, 100);
                FANNHiddenNeurons = (uint)Random.Range(1, 50);
            }
            if (StringLayers)
            {
                FANN = new FANNClass(true, Layers);
            }
            else
            {
                FANN = new FANNClass(true, FANNLayers, 3, FANNHiddenNeurons, 1);
            }
        }
        else
        {
            FANN = new FANNClass();
            FANN.LoadNet(ParentName + NetFileName);
        }

        if (LoadTrainData) FANN.TrainOnData(ParentName + TrainFileName, (uint)TrainCount, Scale, true);

        //FANN = new FANNClass(TrainFile, true, FANNLayers, FANNHiddenNeurons);
        //FANN.TrainOnData(TrainFile, 1000, Scale, true);
        //input = new float[3];
        //output = new float[1];
        //BrainRndMin = new double[3];
        //BrainRndMin = new double[3];
    }

    private float[] Double1dToFloat1d(double[] inputD)
    {
        float[] outputD = new float[inputD.Length];
        for (int i = 0; i < outputD.Length; i++) { outputD[i] = (float)inputD[i]; }
        return outputD;
    }
    private double[] Float1dToDouble1d(float[] inputD)
    {
        double[] outputD = new double[inputD.Length];
        for (int i = 0; i < outputD.Length; i++) { outputD[i] = (double)inputD[i]; }
        return outputD;
    }
    public void CollectData()
    {
        if (!isCollectData)
        {
            TrainListLength = 0;
            TrainEachNum = 0;
            return;
        }
        double[] inputD = Float1dToDouble1d(input);
        double[] outputD = Float1dToDouble1d(output);

        TrainListLength = FANN.AddTrainIOToList(inputD, outputD);
        if (TrainListLength % TrainEachNum == 0)
        {
            MSE = FANN.TrainOnIOList(TrainCount, Scale, true);
            if (ShowBrainLog)
            {
                Debug.Log("MSE=" + MSE);
            }
            SaveTrainIOList();
            SaveNet();
        }
    }

    public void UseBrain()
    {
        if (isUseBrainFunc)
        {
            UseBrainFunc();
            return;
        }
        double[] outputResult = { 1 };
        try
        {
            input = Double1dToFloat1d(FANN.GetNetResultInput(outputResult, BrainError, BrainMaxIterations, BrainRndMin, BrainRndMax, Scale));
        }
        catch { };
        if (ShowBrainLog)
        {
            Debug.Log(FANN.LogResult);
            LogMessage = FANN.LogResult;
        }
    }

    public void UseBrainFunc()
    {
        double[] outputResult = { 1 };
        try
        {
            input = Double1dToFloat1d(FANN.GetNetResultInput(Predict.__Spreadsh_MLP_3_9_2, input.Length, outputResult.Length, outputResult, BrainError, BrainMaxIterations, BrainRndMin, BrainRndMax));
        }
        catch { };
        if (ShowBrainLog)
        {
            Debug.Log(FANN.LogResult);
            LogMessage = FANN.LogResult;
        }
    }

    public void SaveTrainIOList()
    {
        FANN.SaveTrainIOList(ParentName + TrainFileName, false, false);
        FANN.SaveTrainIOListToColumns(ParentName + TrainFileName + "_column");
    }
    public void SaveNet()
    {
        FANN.SaveNet(ParentName + NetFileName, false);
    }

    public void ResultInfo_AppendText()
    {
        string FileResultName = "ResultInfo.txt";
        using (StreamWriter sw = File.AppendText(FileResultName))
        {
            //BrainMaxIterations, BrainError, TrainCount, TrainEachNum, FANNHiddenNeurons
            sw.WriteLine(ParentName + "\t" + ResultInfo + "\t" + FANNHiddenNeurons.ToString() + "\t" + BrainMaxIterations.ToString() + "\t" + BrainError.ToString() + "\t" + TrainCount.ToString() + "\t" + TrainEachNum.ToString() + "\n");
        }
    }
    // Update is called once per frame
    void Update () {
        
    }
}
