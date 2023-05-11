package amlsim;
import java.util.Random;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import amlsim.dists.TruncatedNormal;
import amlsim.dists.TruncatedNormalQuick;

public class TargetedTransactionAmount {

    private SimProperties simProperties;
    private Random random;
    private double target;
    private Boolean isSAR;

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

        // if (this.target == 0.0) {
        //     return this.target;
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
        // RandomGenerator randomGenerator = new JDKRandomGenerator();
        // NormalDistribution normalDistribution = new NormalDistribution(randomGenerator, mean, std);
        // do {
        //     result = normalDistribution.sample();
        //     System.out.println("result: " + result);
        // } while (result < this.target * 0.1 || result > this.target * 0.9);
        
        if (this.target == 0.0) {
            return this.target;
        }
        double mean, std, result;
        if (this.isSAR) {
            mean = simProperties.getMeanTransactionAmountSAR();
            std = simProperties.getStdTransactionAmountSAR();
        }
        else {
            mean = simProperties.getMeanTransactionAmount();
            std = simProperties.getStdTransactionAmount();
        }
        double lb = 10.0; //this.target * 0.1;
        double ub = 5000.0; //this.target * 0.9;
        if (this.target < ub) {
            ub = this.target * 0.9;
        }
        //TruncatedNormalQuick tnq = new TruncatedNormalQuick(mean, std, lb, ub);
        TruncatedNormal tn = new TruncatedNormal(mean, std, lb, ub);
        result = tn.sample();
        return result;
    }
}
