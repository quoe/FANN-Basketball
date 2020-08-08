using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using FANNCSharp;
using FANNCSharp.Double;
using DataType = System.Double;
//using System.Windows.Forms;

namespace FANN_WindowsFormsApplication
{
    class FANNClass
    {
        DataType[] calc_out;
        uint num_layers = 3;
        uint num_input = 2;
        uint num_neurons_hidden = 10;
        uint num_output = 1;
        float desired_error = 0;
        uint max_epochs = 1000;
        uint epochs_between_reports = 10;
        public NeuralNet net;
        public NetworkType NetType;
        public TrainingData TrainData;
        int decimal_point;
        public float ScaleNewInputMin = 0;
        public float ScaleNewInputMax = 1;
        public float ScaleNewOutputMin = 0;
        public float ScaleNewOutputMax = 1;
        public string LogResult;

        public FANNClass()
        {
        }

        /// <summary>
        /// Создание нейросети из файла для тренировки 
        /// </summary>
        /// <param name="DataFilePath">Файл тренировки нейросети формата ".data"</param>
        /// <param name="NetLayerType">Тип сети: true = LAYER, false = SHORTCUT</param>
        /// <param name="num_layers">Количество слоёв</param>
        /// <param name="num_neurons_hidden">Количество нейронов в скрытом слое</param>
        public FANNClass(string DataFilePath, bool NetLayerType, uint num_layers, uint num_neurons_hidden)
        {
            this.TrainData = new TrainingData(DataFilePath);
            SetNetParams(NetLayerType, num_layers, TrainData.InputCount, num_neurons_hidden, TrainData.OutputCount);
            this.net = new NeuralNet(this.NetType, this.num_layers, this.num_input, this.num_neurons_hidden, this.num_output);
        }

        /// <summary>
        /// Создание нейросети
        /// </summary>
        /// <param name="NetLayerType">Тип сети: true = LAYER, false = SHORTCUT</param>
        /// <param name="num_layers">Количество слоёв</param>
        /// <param name="num_input">Количество входов</param>
        /// <param name="num_neurons_hidden">Количество нейронов в скрытом слое</param>
        /// <param name="num_output">Количество выходов</param>
        public FANNClass(bool NetLayerType, uint num_layers, uint num_input, uint num_neurons_hidden, uint num_output)
        {
            SetNetParams(NetLayerType, num_layers, num_input, num_neurons_hidden, num_output);
            this.net = new NeuralNet(this.NetType, this.num_layers, this.num_input, this.num_neurons_hidden, this.num_output);
        }

        private void SetNetParams(bool NetLayerType, uint num_layers, uint num_input, uint num_neurons_hidden, uint num_output)
        {
            this.NetType = NetLayerType ? NetworkType.LAYER : NetworkType.SHORTCUT;
            this.num_layers = num_layers;
            this.num_input = num_input;
            this.num_neurons_hidden = num_neurons_hidden;
            this.num_output = num_output;
        }

        /// <summary>
        /// Установка параметров для тренировки нейросети
        /// </summary>
        /// <param name="ActivationFunctionHidden">Активационная функция скрытого слоя</param>
        /// <param name="ActivationFunctionOutput">Активационная функция выходного нейрона</param>
        /// <param name="TrainStopFunction"></param>
        /// <param name="BitFailLimit"></param>
        /// <param name="TrainingAlgorithm">Алгорит для тренировки</param>
        public void SetNetTrainParams(ActivationFunction ActivationFunctionHidden, ActivationFunction ActivationFunctionOutput, StopFunction TrainStopFunction, double BitFailLimit, TrainingAlgorithm TrainingAlgorithm)
        {
                this.net.ActivationFunctionHidden = ActivationFunctionHidden;//ActivationFunction.SIGMOID_SYMMETRIC;
                this.net.ActivationFunctionOutput = ActivationFunctionOutput;//ActivationFunction.SIGMOID_SYMMETRIC;
                this.net.TrainStopFunction = TrainStopFunction;//StopFunction.STOPFUNC_BIT;
                this.net.BitFailLimit = BitFailLimit;//0.01F;
                this.net.TrainingAlgorithm = TrainingAlgorithm;// TrainingAlgorithm.TRAIN_RPROP;
            
        }

        public void SetScalingParamsValues(float ScaleNewInputMin, float ScaleNewInputMax, float ScaleNewOutputMin, float ScaleNewOutputMax)
        {
            this.ScaleNewInputMin = ScaleNewInputMin;
            this.ScaleNewInputMax = ScaleNewInputMax;
            this.ScaleNewOutputMin = ScaleNewOutputMin;
            this.ScaleNewOutputMax = ScaleNewOutputMax;
        }

        private void SetScaling(TrainingData TrainData)
        {
            net.SetScalingParams(TrainData, this.ScaleNewInputMin, this.ScaleNewInputMax, this.ScaleNewOutputMin, this.ScaleNewOutputMax);
            net.ScaleTrain(this.TrainData);
        }

        /// <summary>
        /// Тренировка нейросети на основе файла
        /// </summary>
        /// <param name="TrainDataFilePath">Путь к файлу с данными для тренировки</param>
        /// <param name="max_epochs">Максимальное количество эпох</param>
        /// <param name="ScaleInput">true = включено масштабирование от 0 до 1, false - исходные данные</param>
        /// <param name="DefaultTrainParams">true = Заполнение параметров тренировки сети по умолачнию</param>
        /// <returns></returns>
        public float TrainOnData(string TrainDataFilePath, uint max_epochs, bool ScaleInput, bool DefaultTrainParams)
        {
            this.TrainData = new TrainingData(TrainDataFilePath);
            this.max_epochs = max_epochs;
            if (DefaultTrainParams)
            {
                SetNetTrainParams(ActivationFunction.SIGMOID_SYMMETRIC, ActivationFunction.SIGMOID_SYMMETRIC, StopFunction.STOPFUNC_BIT, 0.01F, TrainingAlgorithm.TRAIN_RPROP);
            }

            net.InitWeights(TrainData);
            if (ScaleInput) SetScaling(TrainData);

            Debug.WriteLine("Training network on data.");
                net.TrainOnData(TrainData, max_epochs, epochs_between_reports, desired_error);

            DataType[][] input = TrainData.Input;
            DataType[][] output = TrainData.Output;

            string LogStr = "Network TrainOnData test [ ";
            LogResult = LogStr;
            for (int i = 0; i < TrainData.TrainDataLength; i++)
            {
                calc_out = net.Run(input[i]);
                for (int j = 0; j< input[i].Length; j++) //Add input
                {
                    LogResult += input[i][j].ToString() + "; ";
                }
                LogResult += "] -> [ ";
                for (int j = 0; j < calc_out.Length; j++) //Add neuronet output
                {
                    LogResult += calc_out[j].ToString() + "; ";
                }
                LogResult += "], should be [ ";
                for (int j = 0; j < output[i].Length; j++) //Add real output
                {
                    LogResult += output[i][j].ToString() + "; ";
                }
                LogResult += "], difference = [ ";
                for (int j = 0; j < calc_out.Length; j++) //Add neuronet and output difference
                {
                    LogResult += (Math.Abs(calc_out[j] - output[i][j])).ToString() + "; ";
                }
                LogResult += "]\n" + LogStr;
            }
            LogResult += "MSE=" + net.MSE.ToString() + " ]";
            Debug.WriteLine(LogResult);
            return net.MSE;
        }

        public float TrainOnDataEpoch(TrainingData TrainData, int trainCount, bool ScaleInput, bool DefaultTrainParams)
        {
            this.TrainData = TrainData;
            if (DefaultTrainParams)
            {
                SetNetTrainParams(ActivationFunction.SIGMOID_SYMMETRIC, ActivationFunction.SIGMOID_SYMMETRIC, StopFunction.STOPFUNC_BIT, 0.01F, TrainingAlgorithm.TRAIN_RPROP);
            }

            net.InitWeights(TrainData);
            if (ScaleInput) SetScaling(TrainData);

            DataType[][] input = TrainData.Input;
            DataType[][] output = TrainData.Output;

            Debug.WriteLine("Training network epoch on data.");
            for (int i = 0; i < trainCount; i++)
            {
                net.TrainEpoch(TrainData);
                //Debug.WriteLine("Iteration: {0}, MSE: {1}", i + 1, net.MSE);
            }

            string LogStr = "Network TrainOnDataEpoch test [ ";
            LogResult = LogStr;
            for (int i = 0; i < TrainData.TrainDataLength; i++)
            {
                calc_out = net.Run(input[i]);
                for (int j = 0; j < input[i].Length; j++) //Add input
                {
                    LogResult += input[i][j].ToString() + "; ";
                }
                LogResult += "] -> [ ";
                for (int j = 0; j < calc_out.Length; j++) //Add neuronet output
                {
                    LogResult += calc_out[j].ToString() + "; ";
                }
                LogResult += "], should be [ ";
                for (int j = 0; j < output[i].Length; j++) //Add real output
                {
                    LogResult += output[i][j].ToString() + "; ";
                }
                LogResult += "], difference = [ ";
                for (int j = 0; j < calc_out.Length; j++) //Add neuronet and output difference
                {
                    LogResult += (Math.Abs(calc_out[j] - output[i][j])).ToString() + "; ";
                }
                LogResult += "]\n" + LogStr;
            }
            LogResult += "MSE=" + net.MSE.ToString() + " ]";
            Debug.WriteLine(LogResult);
            return net.MSE;
        }

        /// <summary>
        /// Тренировка сети на основе файла и заданного количества эпох, учитывая предыдущий опыт текущего вызова
        /// </summary>
        /// <param name="TrainDataFilePath">Путь к файлу с данными для тренировки</param>
        /// <param name="trainCount">Количество тренировок</param>
        /// <param name="ScaleInput">true = включено масштабирование от 0 до 1, false - исходные данные</param>
        /// <param name="DefaultTrainParams">true = Заполнение параметров тренировки сети по умолачнию</param>
        /// <returns></returns>
        public float TrainOnDataEpoch(string TrainDataFilePath, int trainCount, bool ScaleInput, bool DefaultTrainParams)
        {
            this.TrainData = new TrainingData(TrainDataFilePath);
            return TrainOnDataEpoch(this.TrainData, trainCount, ScaleInput, DefaultTrainParams);
        }

        /// <summary>
        /// Тренировать сеть на основе массивов double[]
        /// </summary>
        /// <param name="input">Входной массив данных</param>
        /// <param name="output">Выходной массив данных</param>
        /// <param name="trainCount">Количество тренировок (циклов)</param>
        /// <param name="DefaultTrainParams">true = Заполнение параметров тренировки сети по умолачнию</param>
        /// <returns></returns>
        public float TrainOnIO(double[] input, double[] output, int trainCount, bool DefaultTrainParams)
        {
            if (DefaultTrainParams)
            {
                SetNetTrainParams(ActivationFunction.SIGMOID_SYMMETRIC, ActivationFunction.SIGMOID_SYMMETRIC, StopFunction.STOPFUNC_BIT, 0.01F, TrainingAlgorithm.TRAIN_RPROP);
            }

            Debug.WriteLine("Training network on IO[].");

            for (int i = 0; i < trainCount; i++)
            {
                net.Train(input, output);
                //Debug.WriteLine("Iteration: {0}, MSE: {1}", i+1, net.MSE);
            }
            string LogStr = "Network TrainOnIO test [ ";
            LogResult = LogStr;
            for (int i = 0; i < TrainData.TrainDataLength; i++)
            {
                calc_out = net.Run(input);
                for (int j = 0; j < input.Length; j++) //Add input
                {
                    LogResult += input[j].ToString() + "; ";
                }
                LogResult += "] -> [ ";
                for (int j = 0; j < calc_out.Length; j++) //Add neuronet output
                {
                    LogResult += calc_out[j].ToString() + "; ";
                }
                LogResult += "], should be [ ";
                for (int j = 0; j < output.Length; j++) //Add real output
                {
                    LogResult += output[j].ToString() + "; ";
                }
                LogResult += "], difference = [ ";
                for (int j = 0; j < calc_out.Length; j++) //Add neuronet and output difference
                {
                    LogResult += (Math.Abs(calc_out[j] - output[j])).ToString() + "; ";
                }
                LogResult += "]\n" + LogStr;
            }
            LogResult += "MSE=" + net.MSE.ToString() + " ]";
            Debug.WriteLine(LogResult);
            return net.MSE;

            calc_out = net.Run(input);
            Debug.WriteLine("Network test ({0},{1}) -> {2}, should be {3}, difference={4}",
                                input[0], input[1], calc_out[0], output[0],
                                Math.Abs(calc_out[0] - output[0]));
            return net.MSE;
        }

        public float TrainOnIO(double[][] input, double[][] output, int trainCount, bool ScaleInput, bool DefaultTrainParams)
        {
            TrainData = new TrainingData();
            TrainData.SetTrainData(input, output);

            net.InitWeights(TrainData);
            if (ScaleInput) SetScaling(TrainData);

            DataType[][] input2 = TrainData.Input;
            DataType[][] output2 = TrainData.Output;

            Debug.WriteLine("Training network on IO[][].");
            for (int i = 0; i < trainCount; i++)
            {
                net.TrainEpoch(TrainData);
                //Debug.WriteLine("Iteration: {0}, MSE: {1}", i + 1, net.MSE);
            }

            string LogStr = "Network TrainOnIO test [ ";
            LogResult = LogStr;
            for (int i = 0; i < TrainData.TrainDataLength; i++)
            {
                calc_out = net.Run(input2[i]);
                for (int j = 0; j < input2[i].Length; j++) //Add input
                {
                    LogResult += input2[i][j].ToString() + "; ";
                }
                LogResult += "] -> [ ";
                for (int j = 0; j < calc_out.Length; j++) //Add neuronet output
                {
                    LogResult += calc_out[j].ToString() + "; ";
                }
                LogResult += "], should be [ ";
                for (int j = 0; j < output2[i].Length; j++) //Add real output
                {
                    LogResult += output2[i][j].ToString() + "; ";
                }
                LogResult += "], difference = [ ";
                for (int j = 0; j < calc_out.Length; j++) //Add neuronet and output difference
                {
                    LogResult += (Math.Abs(calc_out[j] - output2[i][j])).ToString() + "; ";
                }
                LogResult += "]\n" + LogStr;
            }
            LogResult += "MSE=" + net.MSE.ToString() + " ]";
            Debug.WriteLine(LogResult);
            return net.MSE;
        }

        public double[] RunNetOnData(double[] input, bool ScaleInput)
        {
            TrainingData TrainData = new TrainingData();
            double[][] input2 = new double[1][];
            input2[0] = input;
            double[][] output2 = input2;

            TrainData.SetTrainData(input2, output2);
            return RunNetOnData(TrainData, ScaleInput);
        }

        public double[] RunNetOnData(double[] input, double[] output, bool ScaleInput)
        {
            TrainingData TrainData = new TrainingData();
            double[][] input2 = new double[1][];
            input2[0] = input;
            double[][] output2 = new double[1][];
            output2[0] = output;

            TrainData.SetTrainData(input2, output2);
            return RunNetOnData(TrainData, ScaleInput);
        }

        /// <summary>
        /// Выполнить нейросеть на основе данных
        /// </summary>
        /// <param name="TrainData">Переменная с данными</param>
        /// <param name="ScaleInput">true = включено масштабирование от 0 до 1, false - исходные данные</param>
        /// <returns></returns>
        public double[] RunNetOnData(TrainingData TrainData, bool ScaleInput)
        {
            string LogStr = "Network RunNetOnData run [ ";
            LogResult = LogStr;
            for (int i = 0; i < TrainData.TrainDataLength; i++)
            {
                net.ResetMSE();
                if (ScaleInput) net.ScaleInput(TrainData.GetTrainInput((uint)i));
                calc_out = net.Run(TrainData.GetTrainInput((uint)i));
                if (ScaleInput) net.DescaleOutput(calc_out);

                for (int j = 0; j < TrainData.InputAccessor[i].Count; j++) //Add input
                {
                    LogResult += TrainData.InputAccessor[i][j].ToString() + "; ";
                }
                LogResult += "] -> [ ";
                for (int j = 0; j < calc_out.Length; j++) //Add neuronet output
                {
                    LogResult += calc_out[j].ToString() + "; ";
                }
                LogResult += "], should be [ ";
                for (int j = 0; j < TrainData.OutputAccessor[i].Count; j++) //Add real output
                {
                    LogResult += TrainData.OutputAccessor[i][j].ToString() + "; ";
                }
                LogResult += "], difference = [ ";
                for (int j = 0; j < calc_out.Length; j++) //Add neuronet and output difference
                {
                    LogResult += (Math.Abs(calc_out[j] - TrainData.OutputAccessor[i][j])).ToString() + "; ";
                }
                LogResult += "]\n" + LogStr;
            }
                LogResult += "MSE=" + net.MSE.ToString() + " ]";
                Debug.WriteLine(LogResult);
                return calc_out;
        }

        public double[] RunNetOnData(string TrainDataFilePath, bool ScaleInput)
        {
            this.TrainData = new TrainingData(TrainDataFilePath);
            return RunNetOnData(this.TrainData, ScaleInput);
        }

        public double[] RunNetOnData(double[][] input, double[][] output, bool ScaleInput)
        {
            TrainingData TrainData = new TrainingData();
            TrainData.SetTrainData(input, output);
            return RunNetOnData(TrainData, ScaleInput);
        }

        public float TestNetOnData(TrainingData TrainData, bool ScaleInput)
        {
            this.TrainData = TrainData;
            if (ScaleInput) SetScaling(this.TrainData);
            float Test = net.TestData(this.TrainData);
            Console.WriteLine("Test on data result {0}: ", Test);
            return Test;
        }

        public float TestNetOnData(string TrainDataFilePath, bool ScaleInput)
        {
            this.TrainData = new TrainingData(TrainDataFilePath);
            return TestNetOnData(this.TrainData, ScaleInput);
        }

        public float TestNetOnIO(double[] input, double[] output)
        {
            Debug.WriteLine("Test network on IO[].");
            DataType[] Test = new DataType[1];
            Test = net.Test(input, output);
            Console.WriteLine("Run on data result {0} original {1} error {2}", Test[0], output[0],
                                  Math.Abs(Test[0] - output[0]));
            return (float)Test[0];
        }

        public float TestNetOnIO(double[][] input, double[][] output, bool ScaleInput)
        {
            Debug.WriteLine("Test network on IO[][].");
            TrainingData TrainData = new TrainingData();
            TrainData.SetTrainData(input, output);
            if (ScaleInput) SetScaling(TrainData);
            return TestNetOnData(TrainData, ScaleInput);
        }

        public void SaveNet(string NetFilePath, bool SaveToFixed)
        {
            Debug.WriteLine("Saving network.");
            net.Save(NetFilePath);
            if (SaveToFixed)
            {
                decimal_point = net.SaveToFixed(NetFilePath);
                TrainData.SaveTrainToFixed(NetFilePath, (uint)decimal_point);
            }
        }

        public void LoadNet(string NetFilePath) 
        {
            Debug.WriteLine("Load network: " + NetFilePath);
            net = new NeuralNet(NetFilePath);
            net.PrintConnections();
            net.PrintParameters();
            //TrainingData TrainData = new TrainingData("xor.data");
            //RunNetOnData(TrainData, true);
        }
    }
}
