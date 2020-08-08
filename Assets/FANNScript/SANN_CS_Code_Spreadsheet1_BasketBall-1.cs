/**C# deployment code of Neural Networks Model**/

/**==========================================================================
Before running the C# deployment code please read the following.

 STATISTICA variable names will be exported as-is into the C# deployment script;
please verify the resulting script to ensure that the variable names follow the C#
naming conventions and modify the names if necessary.

==========================================================================**/

using System;





public class Predict

{

   public static string __Spreadsh_MLP_3_9_2( double[] ContInputs )

   {

     //"Input Variable" comment is added besides Input(Response) variables.



     int Cont_idx=0;

     double _Input_1__ = ContInputs[Cont_idx++]; //Input Variable

     double _Input_2__ = ContInputs[Cont_idx++]; //Input Variable

     double _Input_3__ = ContInputs[Cont_idx++]; //Input Variable

     string __statist_PredCat="";

    string [] __statist_DCats = new string[2];

    __statist_DCats[0]= "-1";

    __statist_DCats[1]= "1";



    double __statist_ConfLevel=3.0E-300;



    double[] __statist_max_input = new double[3];

    __statist_max_input[0]= 1.32761487960815e+001;

    __statist_max_input[1]= 6.99638122558594e+002;

    __statist_max_input[2]= 6.99932495117188e+002;



    double[] __statist_min_input = new double[3];

    __statist_min_input[0]= 3.37971234321594e+000;

    __statist_min_input[1]= 3.69334220886230e-001;

    __statist_min_input[2]= 3.98707389831543e-001;





    double[,] __statist_i_h_wts = new double[9,3];



    __statist_i_h_wts[0,0]=7.96235041794927e+000;

    __statist_i_h_wts[0,1]=-1.45440198638229e+001;

    __statist_i_h_wts[0,2]=-1.48029704273625e+001;



    __statist_i_h_wts[1,0]=-1.99158893113111e+001;

    __statist_i_h_wts[1,1]=3.20531530396551e+001;

    __statist_i_h_wts[1,2]=3.01751121424909e+000;



    __statist_i_h_wts[2,0]=-2.58951099759686e+000;

    __statist_i_h_wts[2,1]=1.81210375372219e+000;

    __statist_i_h_wts[2,2]=1.90409022645045e+001;



    __statist_i_h_wts[3,0]=-2.26853245625027e+001;

    __statist_i_h_wts[3,1]=1.79812544558069e+001;

    __statist_i_h_wts[3,2]=1.48218546281484e+000;



    __statist_i_h_wts[4,0]=-3.81785109824808e+000;

    __statist_i_h_wts[4,1]=5.02052735627795e+000;

    __statist_i_h_wts[4,2]=1.47465889325606e+001;



    __statist_i_h_wts[5,0]=8.11713601741305e+000;

    __statist_i_h_wts[5,1]=3.28679797566008e+000;

    __statist_i_h_wts[5,2]=-1.19662060637537e+000;



    __statist_i_h_wts[6,0]=-9.82909609954122e+000;

    __statist_i_h_wts[6,1]=3.17254701982197e+000;

    __statist_i_h_wts[6,2]=9.77619838199569e+000;



    __statist_i_h_wts[7,0]=-9.70935333908633e+000;

    __statist_i_h_wts[7,1]=2.48986497134532e+001;

    __statist_i_h_wts[7,2]=9.68548794052845e+000;



    __statist_i_h_wts[8,0]=-1.97397478777446e+001;

    __statist_i_h_wts[8,1]=1.20393886564470e+001;

    __statist_i_h_wts[8,2]=1.21111846593575e+001;



    double[,] __statist_h_o_wts = new double[2,9];



    __statist_h_o_wts[0,0]=4.38562736585139e+000;

    __statist_h_o_wts[0,1]=1.20586422072534e-002;

    __statist_h_o_wts[0,2]=-2.39487854396361e+000;

    __statist_h_o_wts[0,3]=-1.36695058970434e+000;

    __statist_h_o_wts[0,4]=3.21432216762882e+000;

    __statist_h_o_wts[0,5]=-6.04821676302648e-001;

    __statist_h_o_wts[0,6]=-2.01008642659146e-002;

    __statist_h_o_wts[0,7]=2.50087488871914e+000;

    __statist_h_o_wts[0,8]=-3.60317105905397e+000;



    __statist_h_o_wts[1,0]=-4.32969586203894e+000;

    __statist_h_o_wts[1,1]=-3.74935027029792e-002;

    __statist_h_o_wts[1,2]=2.42318024121316e+000;

    __statist_h_o_wts[1,3]=1.38698733849353e+000;

    __statist_h_o_wts[1,4]=-3.22762186156811e+000;

    __statist_h_o_wts[1,5]=6.10729959320623e-001;

    __statist_h_o_wts[1,6]=7.35210844873290e-003;

    __statist_h_o_wts[1,7]=-2.46079131615944e+000;

    __statist_h_o_wts[1,8]=3.68544088526063e+000;



    double[] __statist_hidden_bias = new double[9];

    __statist_hidden_bias[0]=9.07675779061273e+000;

    __statist_hidden_bias[1]=7.12929656519281e+000;

    __statist_hidden_bias[2]=-7.17530866040297e+000;

    __statist_hidden_bias[3]=-1.47955217755254e+001;

    __statist_hidden_bias[4]=-8.81552880856396e+000;

    __statist_hidden_bias[5]=-1.03545479130348e+001;

    __statist_hidden_bias[6]=3.08927516228682e-001;

    __statist_hidden_bias[7]=-1.14436733102810e+001;

    __statist_hidden_bias[8]=-2.63325998038811e+001;



    double[] __statist_output_bias = new double[2];

    __statist_output_bias[0]=4.88943984370608e-001;

    __statist_output_bias[1]=-5.08970202217817e-001;



    double[] __statist_inputs = new double[3];



    double[] __statist_hidden = new double[9];



    double[] __statist_outputs = new double[2];

    __statist_outputs[0] = -1.0e+307;

    __statist_outputs[1] = -1.0e+307;



    __statist_inputs[0]=_Input_1__;

    __statist_inputs[1]=_Input_2__;

    __statist_inputs[2]=_Input_3__;



    double __statist_delta=0;

    double __statist_maximum=1;

    double __statist_minimum=0;

    int __statist_ncont_inputs=3;



    /*scale continuous inputs*/

    for(int __statist_i=0;__statist_i < __statist_ncont_inputs;__statist_i++)

    {

     __statist_delta = (__statist_maximum-__statist_minimum)/(__statist_max_input[__statist_i]-__statist_min_input[__statist_i]);

     __statist_inputs[__statist_i] = __statist_minimum - __statist_delta*__statist_min_input[__statist_i]+ __statist_delta*__statist_inputs[__statist_i];

    }



    int __statist_ninputs=3;

    int __statist_nhidden=9;



    /*Compute feed forward signals from Input layer to hidden layer*/

    for(int __statist_row=0;__statist_row < __statist_nhidden;__statist_row++)

    {

      __statist_hidden[__statist_row]=0.0;

      for(int __statist_col=0;__statist_col < __statist_ninputs;__statist_col++)

      {

       __statist_hidden[__statist_row]= __statist_hidden[__statist_row] + (__statist_i_h_wts[__statist_row,__statist_col]*__statist_inputs[__statist_col]);

      }

     __statist_hidden[__statist_row]=__statist_hidden[__statist_row]+__statist_hidden_bias[__statist_row];

    }



    for(int __statist_row=0;__statist_row < __statist_nhidden;__statist_row++)

    {

      if(__statist_hidden[__statist_row]>100.0)

      {

       __statist_hidden[__statist_row] = 1.0;

      }

      else

      {

       if(__statist_hidden[__statist_row]<-100.0)

       {

        __statist_hidden[__statist_row] = -1.0;

       }

       else

       {

        __statist_hidden[__statist_row] = Math.Tanh(__statist_hidden[__statist_row]);

       }

      }

    }



    int __statist_noutputs=2;



    /*Compute feed forward signals from hidden layer to output layer*/

    for(int __statist_row2=0;__statist_row2 < __statist_noutputs;__statist_row2++)

    {

     __statist_outputs[__statist_row2]=0.0;

    for(int __statist_col2=0;__statist_col2 < __statist_nhidden;__statist_col2++)

      {

       __statist_outputs[__statist_row2]= __statist_outputs[__statist_row2] + (__statist_h_o_wts[__statist_row2,__statist_col2]*__statist_hidden[__statist_col2]);

      }

     __statist_outputs[__statist_row2]=__statist_outputs[__statist_row2]+__statist_output_bias[__statist_row2];

    }





    double __statist_sum=0.0;

    double __statist_maxIndex=0;

    for(int __statist_jj=0;__statist_jj < __statist_noutputs;__statist_jj++)

    {

     if(__statist_outputs[__statist_jj] > 200)

     {

      double __statist_max=__statist_outputs[1];

      __statist_maxIndex=0;

     for(int __statist_ii=0;__statist_ii < __statist_noutputs;__statist_ii++)

    {

      if(__statist_outputs[__statist_ii] > __statist_max)

      {

        __statist_max = __statist_outputs[__statist_ii];

        __statist_maxIndex = __statist_ii;

      }

     }



     for(int __statist_kk=0;__statist_kk < __statist_noutputs;__statist_kk++)

    {

      if(__statist_kk==__statist_maxIndex)

      {

        __statist_outputs[__statist_jj]=1.0;

      }

      else

      {

        __statist_outputs[__statist_kk]=0.0;

      }

     }

    }

    else

    {

     __statist_outputs[__statist_jj] = Math.Exp(__statist_outputs[__statist_jj]);

     __statist_sum = __statist_sum + __statist_outputs[__statist_jj];

    }

   }

     for(int __statist_ll=0;__statist_ll < __statist_noutputs;__statist_ll++)

    {

     if(__statist_sum != 0)

     {

      __statist_outputs[__statist_ll] = __statist_outputs[__statist_ll]/__statist_sum;

     }

    }



    int __statist_PredIndex=1;

    for(int __statist_ii=0;__statist_ii < __statist_noutputs;__statist_ii++)

    {

     if(__statist_ConfLevel < __statist_outputs[__statist_ii])

     {

      __statist_ConfLevel=__statist_outputs[__statist_ii];

      __statist_PredIndex=__statist_ii;

     }

    }



    __statist_PredCat = __statist_DCats[__statist_PredIndex];

        //" Predicted Category = " + __statist_PredCat + " Confidence Level = {0}" + __statist_ConfLevel;
        return __statist_PredCat;
   }



}

