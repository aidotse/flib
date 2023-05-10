package amlsim;
import java.util.Random;

public class TargetedTransactionAmount {

    private SimProperties simProperties;
    private Random random;
    private double target;

    public TargetedTransactionAmount(Number target, Random random) {
        this.simProperties = AMLSim.getSimProp();
        this.random = random;
        this.target = target.doubleValue();
    }

    public double doubleValue() {
        // double minTransactionAmount = simProperties.getMinTransactionAmount();
        // double maxTransactionAmount = simProperties.getMaxTransactionAmount();
        // double min, max, result;
        // if (this.target < maxTransactionAmount) {
        //  max = this.target;
        // }
        // else {
        //  max = maxTransactionAmount;
        // }
        // 
        // if (this.target < minTransactionAmount) {
        //  min = this.target;
        // }
        // else {
        //  min = minTransactionAmount;
        // }
        // 
        // if (max - min <= 0)
        // {
        //  result = this.target;
        // }
        // if (this.target - min <= 100)
        // {
        //  result = this.target;
        // }
        // else
        // {
        //  result =  min + random.nextDouble() * (max - min);
        // }
        
        // double mean = simProperties.getMeanTransactionAmount();
        // double std = simProperties.getStdTransactionAmount();
        // double mu = mean * this.target;
        // double sigma = std * this.target;
        // double result = mu + sigma * random.nextGaussian();
        // if (result >= this.target) {
        //  result = this.target;
        // }
        // if (result <= 0) {
        //  result = 0;
        // }

        double mean = simProperties.getMeanTransactionAmount();
        double std = simProperties.getStdTransactionAmount();
        double result = mean + std * random.nextGaussian();
        if (result >= this.target * 0.9) {
            result = this.target * 0.9;
        }
        if (result <= 0.0) {
            result = this.target * 0.1;
        }
        
        return result;
    }
}
