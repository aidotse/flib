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
        double mean, std, result, lb, ub;
        lb = simProperties.getMinTransactionAmount();
        ub = simProperties.getMaxTransactionAmount();
        if (this.target == 0.0) {
            return this.target;
        }
        if (this.isSAR) {
            mean = simProperties.getMeanTransactionAmountSAR();
            std = simProperties.getStdTransactionAmountSAR();
            lb = 500.0;
            ub = 10000.0;
        }
        else {
            mean = simProperties.getMeanTransactionAmount();
            std = simProperties.getStdTransactionAmount();
        }
        if (this.target < ub) {
            ub = this.target * 0.9;
        }
        //TruncatedNormalQuick tnq = new TruncatedNormalQuick(mean, std, lb, ub);
        TruncatedNormal tn = new TruncatedNormal(mean, std, lb, ub);
        result = tn.sample();
        return result;
    }
}
