//
// Note: No specific bank models are used for this AML typology model class.
//

package amlsim.model.aml;

import java.util.*;

import amlsim.AMLSim;
import amlsim.Account;
import amlsim.TargetedTransactionAmount;

/**
 * Stacked bipartite transactions
 */
public class StackTypology extends AMLTypology {

    List<Account> members;
    private int[] origIdxs;
    private int[] beneIdxs;
    private int numTxs;
    
    private long steps[];
    
    private TargetedTransactionAmount transactionAmount;

    private Random random = AMLSim.getRandom();

    
    // @Override
    // public int getNumTransactions() {
        // int total_members = alert.getMembers().size();
        // int orig_members = total_members / 3; // First 1/3 accounts are originator
        // accounts
        // int mid_members = orig_members; // Second 1/3 accounts are intermediate
        // accounts
        // int bene_members = total_members - orig_members * 2; // Rest of accounts are
        // beneficiary accounts
        // return orig_members * mid_members + mid_members + bene_members;
        // }
        
    StackTypology(double minAmount, double maxAmount, int minStep, int maxStep, int scheduleID, int interval) {
        super(minAmount, maxAmount, minStep, maxStep);

        this.startStep = minStep; //alert.getStartStep();
        this.endStep = maxStep; //alert.getEndStep();
        this.scheduleID = scheduleID; //alert.getScheduleID();
        this.interval = interval; //alert.getInterval();

    }
        
    @Override
    public void setParameters(int modelID) {
        members = alert.getMembers();
        int numMembers = members.size();
        
        numTxs = 0;
        int count = 0;
        int layerSize = random.nextInt(numMembers / 3);
        if (layerSize < 2) {
            layerSize = 2;
        }
        List<Integer> layerSizes = new ArrayList<>();
        layerSizes.add(layerSize);
        count = count + layerSize;
        while (count < numMembers) {
            int nextLayerSize = random.nextInt((numMembers - count) / 2 + 1);
            if (nextLayerSize < layerSize) {
                nextLayerSize = numMembers - count;
            }
            numTxs = numTxs + layerSize * nextLayerSize;
            layerSize = nextLayerSize;
            layerSizes.add(layerSize);
            count = count + layerSize;
        }
        
        // TODO: assert sum

        origIdxs = new int[numTxs];
        beneIdxs = new int[numTxs];
        int numOrigs, numBenes, start, end, increse;
        start = 0;
        increse = 0;
        for (int i = 0; i < layerSizes.size() - 1; i++) {
            numOrigs = layerSizes.get(i);
            numBenes = layerSizes.get(i+1);
            end = start + numOrigs * numBenes;
            for (int j = 0; j < end - start; j++) {
                origIdxs[j + start] = j / numBenes + increse;
                beneIdxs[j + start] = j % numBenes + numOrigs + increse;
            }
            start = end;
            increse = increse + numOrigs;
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
        // TODO: shuffle steps
        start = 0;
        for (int i = 0; i < layerSizes.size()-1; i++) {
            end = start + layerSizes.get(i) * layerSizes.get(i+1);
            for (int j = start; j < end; j++) {
                int randomIndexToSwap = random.nextInt(end - start) + start;
			    long temp = steps[randomIndexToSwap];
			    //int temp1 = origIdxs[randomIndexToSwap];
                //int temp2 = beneIdxs[randomIndexToSwap];
                steps[randomIndexToSwap] = steps[j];
                //origIdxs[randomIndexToSwap] = origIdxs[j];
                //beneIdxs[randomIndexToSwap] = beneIdxs[j];
                steps[j] = temp;
                //origIdxs[j] = temp1;
                //beneIdxs[j] = temp2;
			}
            start = end;
        }
    }

    @Override
    public String getModelName() {
        return "StackTypology";
    }

    @Override
    public void sendTransactions(long step, Account acct) {

        //int total_members = alert.getMembers().size();
        //int orig_members = total_members / 3; // First 1/3 accounts are originator accounts
        //int mid_members = orig_members; // Second 1/3 accounts are intermediate accounts
        //int bene_members = total_members - orig_members * 2; // Rest of accounts are beneficiary accounts
        //
        //for (int i = 0; i < orig_members; i++) { // originator accounts --> Intermediate accounts
        //    Account orig = alert.getMembers().get(i);
        //    //if (!orig.getID().equals(acct.getID())) {
        //    //    continue;
        //    //}
        //
        //    int numBene = (orig_members + mid_members) - orig_members;
        //    TargetedTransactionAmount transactionAmount = getTransactionAmount(numBene, orig.getBalance());
        //
        //    for (int j = orig_members; j < (orig_members + mid_members); j++) {
        //        Account bene = alert.getMembers().get(j);
        //        makeTransaction(step, transactionAmount.doubleValue(), orig, bene, AMLTypology.STACK);
        //    }
        //}
        //
        //for (int i = orig_members; i < (orig_members + mid_members); i++) { // Intermediate accounts --> Beneficiary
        //                                                                    // accounts
        //    Account orig = alert.getMembers().get(i);
        //    //if (!orig.getID().equals(acct.getID())) {
        //    //    continue;
        //    //}
        //
        //    int numBene = total_members - (orig_members + mid_members);
        //    TargetedTransactionAmount transactionAmount = getTransactionAmount(numBene, orig.getBalance());
        //
        //    for (int j = (orig_members + mid_members); j < total_members; j++) {
        //        Account bene = alert.getMembers().get(j);
        //        makeTransaction(step, transactionAmount.doubleValue(), orig, bene, AMLTypology.STACK);
        //    }
        //}

        for (int i = 0; i < numTxs; i++) {
            if (step == steps[i]) {
                Account orig = members.get(origIdxs[i]);
                Account bene = members.get(beneIdxs[i]);
                transactionAmount = new TargetedTransactionAmount(orig.getBalance(), random, true);
                makeTransaction(step, transactionAmount.doubleValue(), orig, bene, alert.isSAR(), alert.getAlertID(), AMLTypology.STACK);
            }
        }
    }

    private TargetedTransactionAmount getTransactionAmount(int numBene, double origBalance) {
        if (numBene == 0) {
            return new TargetedTransactionAmount(0, random, true);
        }
        return new TargetedTransactionAmount(origBalance / numBene, random, true);
    }
}
