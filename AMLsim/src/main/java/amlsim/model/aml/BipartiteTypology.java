//
// Note: No specific bank models are used for this AML typology model class.
//

package amlsim.model.aml;

import amlsim.AMLSim;
import amlsim.Account;
import amlsim.TargetedTransactionAmount;

import java.util.*;

/**
 * Bipartite transaction model
 * Some accounts send money to a different account set
 */
public class BipartiteTypology extends AMLTypology {

    List<Account> members;
    private int[] origIdxs;
    private int[] beneIdxs;

    private int numOrigs;
    private int numBenes;
    private int numTxs;
    
    private long[] steps;
    private double[] amounts;
    
    private Random random = AMLSim.getRandom();
    
    public BipartiteTypology(double minAmount, double maxAmount, int minStep, int maxStep, int scheduleID, int interval, String sourceType) {
        super(minAmount, maxAmount, minStep, maxStep, sourceType);
        
        this.startStep = minStep; //alert.getStartStep();
        this.endStep = maxStep; //alert.getEndStep();
        this.scheduleID = scheduleID; //alert.getScheduleID();
        this.interval = interval; //alert.getInterval();

    }

    @Override
    public void setParameters(int modelID) {
        // Set members
        members = alert.getMembers();
        int numMembers = members.size();
        numOrigs = numMembers / 2; // TODO: make random
        numBenes = numMembers - numOrigs;
        
        numTxs = numOrigs * numBenes;
        origIdxs = new int[numTxs];
        beneIdxs = new int[numTxs];

        for (int i = 0; i < numTxs; i++){
            origIdxs[i] = i / numBenes;
            beneIdxs[i] = i % numBenes + numOrigs;
        }

        // Set transaction schedule
        int range = (int) (this.endStep - this.startStep + 1);// get the range of steps
        steps = new long[numTxs];
        if (scheduleID == FIXED_INTERVAL) {
            if (interval * numTxs > range) { // if needed modifies interval to make time for all txs
                interval = range / numTxs;
            }
            for (int i = 0; i < numTxs; i++) {
                steps[i] = startStep + interval * i;
            }
        } else if (scheduleID == RANDOM_INTERVAL) {
            interval = generateFromInterval(range / numTxs) + 1;
            for (int i = 0; i < numTxs; i++) {
                steps[i] = startStep + interval * i;
            }
        } else if (scheduleID == UNORDERED) {
            for (int i = 0; i < numTxs; i++) {
                steps[i] = generateFromInterval(range) + startStep;
            }
        } else if (scheduleID == SIMULTANEOUS || range < 2) {
            long step = generateFromInterval(range) + this.startStep;
            Arrays.fill(steps, step);
        }

        // Set transaction amounts
        amounts = new double[numTxs];
        for (int i = 0; i < numTxs; i++) {
            TargetedTransactionAmount transactionAmount= new TargetedTransactionAmount(100000, random, true); // TODO: Handle max illicit fund init 
            amounts[i] = transactionAmount.doubleValue();
        }
        
    }

    @Override
    public String getModelName() {
        return "BipartiteTypology";
    }

    @Override
    public void sendTransactions(long step, Account acct) {
        if (step == this.stepReciveFunds) {
            for (int i = 0; i < numTxs; i++) {
                Account orig = members.get(origIdxs[i]);
                if (this.sourceType.equals("CASH")) {
                    acct.depositCash(amounts[i]);
                } else if (this.sourceType.equals("TRANSFER")){
                    AMLSim.handleIncome(step, "TRANSFER", amounts[i], orig, false, (long) -1, (long) 0);
                }
            }
        }
        for (int i = 0; i < numTxs; i++) {
            if (step == steps[i]) {
                Account orig = members.get(origIdxs[i]);
                Account bene = members.get(beneIdxs[i]);
                double amount = amounts[i] * (1.0 - marginRatio);
                if (!isValidAmount(amount, orig)) {
                    amount = 0.9 * orig.getBalance();
                }
                makeTransaction(step, amount, orig, bene, alert.isSAR(), alert.getAlertID(), AMLTypology.BIPARTITE);
            }
        }
    }
}
