/*
   DO NOT START CODING THIS FILE UNTIL YOU HAVE COMPLETED AddOneNNTester.java!
*/

import java.util.*;

/**
   A neural network which is able to add one to a number. 
   Sample input: 1; sample output: 2.
*/
class AddOneNN {
   /* Declare instance variables: 
      an NNLayer for your hidden layer; 
      an NNNode for your output node; and
      a double for your learning rate */
	
	private NNLayer hiddenLayer;
	private NNNode output;
	private double learningRate;

    public AddOneNN(int numHiddenNodes, double learningRate) {
      /* Initialize the three instance variables.
         The hidden layer should have one incoming weight per node.
         The output node should have numHiddenNodes incoming weights per node.
        */
    	this.learningRate = learningRate;
    	this.hiddenLayer = new NNLayer(numHiddenNodes,1);
    	this.output = new NNNode(numHiddenNodes);
    }

    public double test(Map<Double, Double> testData) {
      /*
         Goal: run each of our test inputs through the network. calculate accuracy of outputs via root mean squared error.
         
         start an error variable at zero. this will calculate the sum of squared errors.
         for each key/value pair in the test data map:
            "feed forward": run the input through the neural network to get the output.
                  reminder: hidden node activation = activationFunction(bias + weight * input)
                      output = bias + weight * activation for hidden node #1 + weight * activation for hidden node #2 + ...
            calculate the squared error: (predicted output - correct output)^2. add this to your error variable.
            optional: for debugging/learning purposes, print your input, expected output, and predicted output
         
         return the error: square root of (your error sum divided by the number of inputs in your test data set)         
      */
    	
    double error = 0.0;
    for (Double key : testData.keySet()) {
    	double activation = 0.0;
    	double numOutput = 0.0;
    	List<NNNode> layer = hiddenLayer.getNodes();
    	int pos = 0;
    	for (NNNode node : layer) {
    		activation = 0.0;
    		activation = node.getBias() + node.getWeight(0) * key;
    		numOutput += activation*this.output.getWeight(pos);
    		pos++;
    		
    	}
    	numOutput +=  this.output.getBias();
    	System.out.println("INPUT: " + key);
        System.out.println("OUTPUT: " + numOutput);
    	error += Math.pow(numOutput-key, 2);
    }
      
      return Math.sqrt(error/testData.size()); 
    }
    

    public void train(Map<Double, Double> trainingData, int epochs) {
    	    double error;
    	    double[] hiddenActivations;

    	    for (int epoch = 0; epoch < epochs; epoch++) {
    	        error = 0.0;

    	        for (Double input : trainingData.keySet()) {
    	            double expectedOutput = trainingData.get(input);

    	            // Feedforward step
    	            hiddenActivations = new double[hiddenLayer.getNodes().size()];
    	            double numOutput = 0.0;

    	            List<NNNode> layer = hiddenLayer.getNodes();
    	            int pos = 0;
    	            for (NNNode node : layer) {
    	                double activation = node.getBias() + node.getWeight(0) * input;
    	                hiddenActivations[pos] = activation;
    	                numOutput += activation * this.output.getWeight(pos);
    	                pos++;
    	            }

    	            numOutput += this.output.getBias();
    	            
    	            // Calculate error
    	            double outputError = expectedOutput - numOutput;
    	            error += outputError * outputError;

    	            // Backpropagation step
    	            // Update output node weights and bias
    	            this.output.setBias(this.output.getBias() + learningRate * outputError);

    	            pos = 0;
    	            for (NNNode node : layer) {
    	                double updatedWeight = this.output.getWeight(pos) + learningRate * outputError * hiddenActivations[pos];
    	                this.output.setWeight(pos, updatedWeight);
    	                pos++;
    	            }

    	            // Update hidden layer nodes' weights and biases
    	            pos = 0;
    	            for (NNNode node : layer) {
    	                double nodeError = outputError * this.output.getWeight(pos);

    	                // Update weight from input to hidden node
    	                double updatedHiddenWeight = node.getWeight(0) + learningRate * nodeError * input;
    	                node.setWeight(0, updatedHiddenWeight);

    	                // Update bias for hidden node
    	                double updatedHiddenBias = node.getBias() + learningRate * nodeError;
    	                node.setBias(updatedHiddenBias);
    	                pos++;
    	            }
    	        }

    	    }
    	}

    	
       /* DO THIS METHOD LAST! Check in with your teacher before starting this method. */
       
       /*
            for each epoch:
               for each key/value pair in the training data map:
                  "feed forward": run the input through the neural network to get the output. HOWEVER,
                     this version, unlike your test() version, should save the hidden activations information -
                     we will need them later. consider storing them in a double[] hiddenActivations.
                     
                   "backpropogate": go backwards through the network (starting at output) and adjust weights/biases:
                     (note: I'm recommending you do multiple loops for this. It could technically be done with one loop,
                              but doing it with multiple is much more extensible for if you want to add more hidden layers later.)
         
                     first, update the output node's weights and bias:
                     calculate the output error (expected output - actual output from the feed forward) 
                     update the output node's bias: bias = bias + output error * learning rate
                     for each hidden layer node:
                         set the weight between that node and the output to: its previous weight + output error * learning rate * activation for that hidden node
                         

                     then, update the hidden layer node's weights and biases:
                     for each hidden layer node:
                         calculate the node's error: output error * weight from hidden node to output.
                         update the weight from input to this node to: previous weight + learning rate * node's error * input
                         update the bias for this node to: previous bias + learning rate * node's error
             
       */

    

    
    
    
   
   /*
      Done with training and testing?  Check with your teacher to see if it works correctly!
      
      Then, try some optional challenges:
      
      1. Adjust your "hyperparameters"! What happens if you increase or decrease: 
            The learning rate? 
            The number of neurons in the hidden layer?
            The number of epochs?
            The way biases and weights are initialized in NNNode.java?
         More specifically, can you get your test error down to 0.5? 0.1? 0.01?
      
      2. Add an activation function. Google (or ask ChatGPT) for more information!
         Note: this requires some calculus to fully understand, but you can use it
               without understanding the calculus part.
   */
    
}