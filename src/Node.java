import java.util.*;

/**
 * Class for internal organization of a Neural Network. There are 5 types of
 * nodes. Check the type attribute of the node for details. Feel free to modify
 * the provided function signatures to fit your own implementation
 */

public class Node {
	private int type = 0; // 0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
	public ArrayList<NodeWeightPair> parents = null; // Array List that will
														// contain the parents
														// (including the bias
														// node) with weights if
														// applicable

	private double inputValue = 0.0;
	private double outputValue = 0.0;
	private double outputGradient = 0.0;
	private double delta = 0.0; // input gradient

	// Create a node with a specific type
	Node(int type) {
		if (type > 4 || type < 0) {
			System.out.println("Incorrect value for node type");
			System.exit(1);

		} else {
			this.type = type;
		}

		if (type == 2 || type == 4) {
			parents = new ArrayList<>();
		}
	}

	// For an input node sets the input value which will be the value of a
	// particular attribute
	public void setInput(double inputValue) {
		if (type == 0) { // If input node
			this.inputValue = inputValue;
		}
	}

	/**
	 * Calculate the output of a node. You can get this value by using
	 * getOutput()
	 */
	public void calculateOutput(ArrayList<Node> outputNodes) {
		if (type == 2 || type == 4) { // Not an input or bias node
			double sum = 0.0;
			for (NodeWeightPair p : parents) {
				sum += p.weight * p.node.getOutput();
			}
			if (type == 2) { // if hidden node use ReLU
				outputValue = Math.max(0, sum);
			} else { // if output node use Softmax
				double deno = calculateDeno(outputNodes);
				outputValue = Math.exp(sum) / deno;
			}
		}
	}

	// Gets the output value
	public double getOutput() {
		if (type == 0) { // Input node
			return this.inputValue;
		} else if (type == 1 || type == 3) { // Bias node
			return 1.00;
		} else {
			return outputValue;
		}
	}

	// Calculate the delta value of a node.
	public void calculateDelta(int target, ArrayList<Node> outputNodes, int index) {
		if (type == 2 || type == 4) {
			if (type == 4) { // output node
				delta = (double) target - this.getOutput();
			} else { // hidden node
				double sum = 0.0;
				for (NodeWeightPair p : parents) {
					sum += p.weight * p.node.getOutput();
				}
				if (sum > 0) {
					sum = 0;
					for (Node o : outputNodes) {
						sum += o.parents.get(index).weight * o.delta;
					}
					this.delta = sum;
				}
			}
		}
	}

	// Update the weights between parents node and current node
	public void updateWeight(double learningRate) {
		if (type == 2 || type == 4) {
			for (NodeWeightPair p : parents) {
				p.weight += learningRate * delta * p.node.getOutput();
			}
		}
	}

	// calculate denominator in softmax
	private double calculateDeno(ArrayList<Node> outputNodes) {
		double deno = 0.0;
		// weighted sum of the inputs for each hidden node
		double hSum;
		for (Node o : outputNodes) {
			hSum = 0;
			for (NodeWeightPair pp : o.parents) {
				hSum += pp.weight * pp.node.getOutput();
			}
			deno += Math.exp(hSum);
		}
		return deno;
	}
}
