package amlsim;
import java.util.Random;

public class TargetedTransactionAmount {

    private SimProperties simProperties;
    private Random random;
    private double target;
    private Boolean isSAR = false;

    // public TargetedTransactionAmount(Number target, Random random) {
    //     this.simProperties = AMLSim.getSimProp();
    //     this.random = random;
    //     this.target = target.doubleValue();
    // }

    public TargetedTransactionAmount(Number target, Random random, Boolean isSAR) {
        this.simProperties = AMLSim.getSimProp();
        this.random = random;
        this.target = target.doubleValue();
        this.isSAR = isSAR;
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

        // double mean, std, result;
        // if (this.isSAR) {
        //     mean = simProperties.getMeanTransactionAmountSAR();
        //     std = simProperties.getStdTransactionAmountSAR();
        // }
        // else {
        //     mean = simProperties.getMeanTransactionAmount();
        //     std = simProperties.getStdTransactionAmount();
        // }
        // result = mean + std * random.nextGaussian();
        // if (result >= this.target * 0.9) {
        //     result = this.target * 0.9;
        // }
        // if (result <= 0.0) {
        //     result = this.target * 0.1;
        // }

        double mean, std, result;
        if (this.isSAR) {
            mean = simProperties.getMeanTransactionAmountSAR();
            std = simProperties.getStdTransactionAmountSAR();
            do {
                result = mean + std * random.nextGaussian();
            } while (result < this.target * 0.1 || result > this.target * 0.9);
        }
        else {
            mean = simProperties.getMeanTransactionAmount();
            std = simProperties.getStdTransactionAmount();
            do {
                result = mean + std * random.nextGaussian();
            } while (result < this.target * 0.1 || result > this.target * 0.9);
        }
        if (this.target == 0.0) {
            result = 0.0;
        }
        return result;
    }
}
