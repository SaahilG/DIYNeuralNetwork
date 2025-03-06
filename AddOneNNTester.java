import java.util.*;

public class AddOneNNTester
{
   public static void main(String[] args) {
        /*
         Create an AddOneNN object with 2 hidden nodes, and a learning rate of 10^-6.
         
         Create a Map<Double, Double> for your training data. It should map a valid input to its correct output.
         Add input data from -1000 to 1000 to your training data set. It should contain (-1000, -999) through (1000, 1001)
         Create a Map<Double, Double> for your testing data. It should map a valid input to its correct output.
         Add input data from 1000 to 2000 to your testing data set. It should contain (1000, 1001) through (2000, 2001)
         
         Call the train method to train your neural network with 20 epochs.
         Call the test method to test your neural network, and print its return value (the root mean squared error).
         */
         
         /*
            Testing your code:
            Before modifying AddOneNN, this should print zero.
            After completing everything except the train() function in AddOneNN, this should print a large error, around 1500.
            After completing the entirety of AddOneNN, this should print a small error, around 0.2.
         */
	   
	   AddOneNN NN = new AddOneNN(400, 0.0000001);
	   Map<Double, Double> trainingData = new HashMap<Double, Double>();
	   
	   for(int i = 0; i <= 100; i++) {
		   trainingData.put((double)i, (double)i*i);
	   }
	   
	   Map<Double, Double> testData = new HashMap<Double, Double>();
	   
	   for(int i = 100; i <= 200; i++) {
		   testData.put((double)i, (double)i*i);
	   }
	   
	   NN.train(trainingData, 20000);
	   System.out.println(NN.test(testData));
	  
   }
}