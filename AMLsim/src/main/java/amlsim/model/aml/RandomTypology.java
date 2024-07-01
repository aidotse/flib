//
// Note: No specific bank models are used for this AML typology model class.
//

package amlsim.model.aml;

import amlsim.AMLSim;
import amlsim.Account;
import amlsim.TargetedTransactionAmount;

import java.util.*;

/**
 * The main account makes a transaction with one of the neighbor accounts
 * and the neighbor also makes transactions with its neighbors.
 * The beneficiary account and amount of each transaction are determined
 * randomly.
 */
public class RandomTypology extends AMLTypology { // TODO: review this pattern, not realistic? similar to cycle but with random amounts

    private static Random random = AMLSim.getRandom();
    private Set<Long> steps = new HashSet<>(); // Set of simulation steps when the transaction is performed
    private Account nextOrig; // Originator account for the next transaction

    @Override
    public void setParameters(int modelID) {
        int numMembers = alert.getMembers().size();
        for (int i = 0; i < numMembers; i++) {
            steps.add(getRandomStep());
        }
        nextOrig = alert.getMainAccount();
    }

    // @Override
    // public int getNumTransactions() {
    // return alert.getMembers().size();
    // }

    RandomTypology(double minAmount, double maxAmount, int minStep, int maxStep, String sourceType) {
        super(minAmount, maxAmount, minStep, maxStep, sourceType);
    }

    @Override
    public String getModelName() {
        return "RandomTypology";
    }

    public boolean isValidStep(long step) {
        return super.isValidStep(step) && steps.contains(step);
    }

    public void sendTransactions(long step, Account acct) {
        boolean isSAR = alert.isSAR();
        long alertID = alert.getAlertID();
        if (!isValidStep(step))
            return;
        
        if (step == this.stepReciveFunds) {
            TargetedTransactionAmount transactionAmount = new TargetedTransactionAmount(100000, random, true); // TODO: Handle max illicit fund init 
            if (this.sourceType.equals("CASH")) {
                nextOrig.depositCash(transactionAmount.doubleValue());
            } else if (this.sourceType.equals("TRANSFER")) {
                AMLSim.handleIncome(step, "TRANSFER", transactionAmount.doubleValue(), nextOrig, true, (long) -1, (long) 0, "OTHER");
            }
        }
        
        List<Account> beneList = nextOrig.getBeneList();
        int numBenes = beneList.size();
        if (numBenes == 0)
            return;

        int idx = random.nextInt(numBenes);
        Account bene = beneList.get(idx);

        TargetedTransactionAmount transactionAmount = new TargetedTransactionAmount(nextOrig.getBalance(), random,
                true);
        makeTransaction(step, transactionAmount.doubleValue(), nextOrig, bene, isSAR, (int) alertID,
                AMLTypology.RANDOM); // Main account makes transactions to one of the neighbors
        nextOrig = bene; // The next originator account is the previous beneficiary account
    }
}
