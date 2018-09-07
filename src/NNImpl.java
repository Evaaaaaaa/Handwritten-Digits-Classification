import java.util.*;

/**
 * The main class that handles the entire network Has multiple attributes each
 * with its own use
 */

public class NNImpl {
	private ArrayList<Node> inputNodes; // list of the output layer nodes.
	private ArrayList<Node> hiddenNodes; // list of the hidden layer nodes
	private ArrayList<Node> outputNodes; // list of the output layer nodes
	private ArrayList<Instance> trainingSet; // the training set
	private double learningRate; // variable to store the learning rate
	private int maxEpoch; // variable to store the maximum number of epochs
	private Random random; // random number generator to shuffle the training
							// set

	/**
	 * This constructor creates the nodes necessary for the neural network Also
	 * connects the nodes of different layers After calling the constructor the
	 * last node of both inputNodes and hiddenNodes will be bias nodes.
	 */

	NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random,
			Double[][] hiddenWeights, Double[][] outputWeights) {
		this.trainingSet = trainingSet;
		this.learningRate = learningRate;
		this.maxEpoch = maxEpoch;
		this.random = random;

		// input layer nodes
		inputNodes = new ArrayList<>();
		int inputNodeCount = trainingSet.get(0).attributes.size();
		int outputNodeCount = trainingSet.get(0).classValues.size();
		for (int i = 0; i < inputNodeCount; i++) {
			Node node = new Node(0);
			inputNodes.add(node);
		}

		// bias node from input layer to hidden
		Node biasToHidden = new Node(1);
		inputNodes.add(biasToHidden);

		// hidden layer nodes
		hiddenNodes = new ArrayList<>();
		for (int i = 0; i < hiddenNodeCount; i++) {
			Node node = new Node(2);
			// Connecting hidden layer nodes with input layer nodes
			for (int j = 0; j < inputNodes.size(); j++) {
				NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		// bias node from hidden layer to output
		Node biasToOutput = new Node(3);
		hiddenNodes.add(biasToOutput);

		// Output node layer
		outputNodes = new ArrayList<>();
		for (int i = 0; i < outputNodeCount; i++) {
			Node node = new Node(4);
			// Connecting output layer nodes with hidden layer nodes
			for (int j = 0; j < hiddenNodes.size(); j++) {
				NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}
			outputNodes.add(node);
		}
	}

	/**
	 * Get the prediction from the neural network for a single e Return the
	 * index with highest output values. For example if the outputs of the
	 * outputNodes are [0.1, 0.5, 0.2], it should return 1. The parameter is a
	 * single e
	 */

	public int predict(Instance e) {
		// max output
		double max = 0.0;
		// the index of the max output
		int idx = 0;

		for (int i = 0; i < inputNodes.size() - 1; i++) {
			inputNodes.get(i).setInput(e.attributes.get(i));
		}
		for (Node h : hiddenNodes) {
			h.calculateOutput(outputNodes);
		}
		for (int i = 0; i < outputNodes.size(); i++) {
			outputNodes.get(i).calculateOutput(outputNodes);
			if ((outputNodes.get(i).getOutput() > max)) {
				max = outputNodes.get(i).getOutput();
				idx = i;
			}
		}
		return idx;
	}

	/**
	 * Train the neural networks with the given parameters
	 * <p>
	 * The parameters are stored as attributes of this class
	 */

	public void train() {
		double totalLoss;
		for (int i = 0; i < maxEpoch; i++) {
			Collections.shuffle(trainingSet, random);
			totalLoss = 0.0;
			for (Instance e : trainingSet) {
				for (int j = 0; j < inputNodes.size() - 1; j++) {
					inputNodes.get(j).setInput(e.attributes.get(j));
				}
				for (int j = 0; j < hiddenNodes.size(); j++) {
					hiddenNodes.get(j).calculateOutput(outputNodes);
				}
				for (int j = 0; j < outputNodes.size(); j++) {
					outputNodes.get(j).calculateOutput(outputNodes);
					// calculate delta for output nodes
					outputNodes.get(j).calculateDelta(e.classValues.get(j), outputNodes, 0);
				}
				// calculate delta for hidden nodes
				for (int j = 0; j < hiddenNodes.size(); j++) {
					hiddenNodes.get(j).calculateDelta(0, outputNodes, j);
				}
				// updateWeight
				for (int j = 0; j < outputNodes.size(); j++) {
					outputNodes.get(j).updateWeight(learningRate);
				}
				for (int j = 0; j < hiddenNodes.size(); j++) {
					hiddenNodes.get(j).updateWeight(learningRate);
				}
			}
			for (Instance e : trainingSet) {
				totalLoss += loss(e);
			}
			totalLoss = totalLoss / trainingSet.size();
			System.out.printf("Epoch: " + i + ", Loss: %.8e\n", totalLoss);
		}
	}

	/**
	 * Calculate the cross entropy loss from the neural network for a single
	 * Instance. The parameter is a single Instance
	 */
	private double loss(Instance e) {
		double loss = 0.0;
		predict(e);
		for (int i = 0; i < outputNodes.size(); i++) {
			loss -= e.classValues.get(i) * Math.log(outputNodes.get(i).getOutput());
		}
		return loss;
	}
}
