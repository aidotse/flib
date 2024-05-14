//
// Note: No specific bank models are used for this AML typology model class.
//

package amlsim.model.aml;

import amlsim.AMLSim;
import amlsim.Account;
import amlsim.TargetedTransactionAmount;

import java.util.*;

/**
 * The main account distributes money to multiple members
 */
public class FanOutTypology extends AMLTypology {

    // Originator and beneficiary accounts
    private Account orig;
    private List<Account> beneList = new ArrayList<>();

    private Random random = AMLSim.getRandom();

    private long[] steps;
    private double[] amounts;
    
    FanOutTypology(double minAmount, double maxAmount, int minStep, int maxStep, String sourceType) {
        super(minAmount, maxAmount, minStep, maxStep, sourceType);
    }

    public int getNumTransactions() {
        return alert.getMembers().size() - 1;
    }

    public void setParameters(int scheduleID) {
        // Set members
        List<Account> members = alert.getMembers();
        Account mainAccount = alert.getMainAccount();
        orig = mainAccount != null ? mainAccount : members.get(0);
        for (Account bene : members) {
            if (orig != bene)
                beneList.add(bene);
        }

        int numBenes = beneList.size();
        
        steps = new long[numBenes];
        if (scheduleID == SIMULTANEOUS) {
            long step = getRandomStep();
            Arrays.fill(steps, step);
        } else if (scheduleID == FIXED_INTERVAL) {
            int range = (int) (endStep - startStep + 1);
            if (numBenes < range) {
                interval = range / numBenes;
                for (int i = 0; i < numBenes; i++) {
                    steps[i] = startStep + interval * i;
                }
            } else {
                long batch = numBenes / range;
                for (int i = 0; i < numBenes; i++) {
                    steps[i] = startStep + i / batch;
                }
            }
        } else if (scheduleID == RANDOM_INTERVAL || scheduleID == UNORDERED) {
            for (int i = 0; i < numBenes; i++) {
                steps[i] = getRandomStep();
            }
        }

        // Set transaction amounts
        amounts = new double[numBenes];
        for (int i = 0; i < numBenes; i++) {
            TargetedTransactionAmount transactionAmount= new TargetedTransactionAmount(100000, random, true); // TODO: Handle max illicit fund init 
            amounts[i] = transactionAmount.doubleValue();
        }
    }

    @Override
    public String getModelName() {
        return "FanOutTypology";
    }

    @Override
    public void sendTransactions(long step, Account acct) {
        if (!orig.getID().equals(acct.getID())) {
            return;
        }
        long alertID = alert.getAlertID();
        boolean isSAR = alert.isSAR();
        
        if (step == this.stepReciveFunds) {
            for (int i = 0; i < beneList.size(); i++) {
                if (this.sourceType.equals("CASH")) {
                    acct.depositCash(amounts[i]);
                } else if (this.sourceType.equals("TRANSFER")){
                    AMLSim.handleIncome(step, "TRANSFER", amounts[i], orig, true, (long) -1, (long) 0, "OTHER");
                }
            }
        }

        for (int i = 0; i < beneList.size(); i++) {
            if (steps[i] == step) {
                Account bene = beneList.get(i);
                double amount = amounts[i] * (1.0 - marginRatio);
                if (!isValidAmount(amount, orig)) {
                    amount = 0.9 * orig.getBalance();
                }
                makeTransaction(step, amount, orig, bene, isSAR, alertID, AMLTypology.AML_FAN_OUT);
            }
        }
    }

    // TODO: remove?
    // private TargetedTransactionAmount getTransactionAmount() {
    //     if (this.beneList.size() == 0) {
    //         return new TargetedTransactionAmount(0, this.random, true);
    //     }
    //     return new TargetedTransactionAmount(orig.getBalance() / this.beneList.size(), random, true);
    // }
}
