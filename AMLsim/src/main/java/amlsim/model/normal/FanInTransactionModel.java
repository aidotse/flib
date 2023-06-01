package amlsim.model.normal;

import amlsim.Account;
import amlsim.AccountGroup;
import amlsim.TargetedTransactionAmount;
import amlsim.model.AbstractTransactionModel;

import java.util.*;

/**
 * Receive money from one of the senders (fan-in)
 */
public class FanInTransactionModel extends AbstractTransactionModel {

    // Originators and the main beneficiary
    private Account bene; // The destination (beneficiary) account
    private List<Account> origList = new ArrayList<>(); // The origin (originator) accounts

    private long[] steps;

    private Random random;
    private TargetedTransactionAmount transactionAmount;

    public FanInTransactionModel(
            AccountGroup accountGroup,
            Random random) {
        this.accountGroup = accountGroup;
        this.random = random;
    }

    public void setParameters(int interval, long start, long end) {
        super.setParameters(interval, start, end);
        if (this.startStep < 0) { // decentralize the first transaction step
            this.startStep = generateStartStep(interval);
        }

        // Set members
        List<Account> members = accountGroup.getMembers();
        Account mainAccount = accountGroup.getMainAccount();
        bene = mainAccount != null ? mainAccount : members.get(0); // The main account is the beneficiary
        for (Account orig : members) { // The rest of accounts are originators
            if (orig != bene)
                origList.add(orig);
        }
        
        // Set transaction schedule
        int schedulingID = this.accountGroup.getScheduleID();
        int numOrigs = origList.size();
        
        steps = new long[numOrigs];

        int range = (int) (end - start + 1);// get the range of steps

        if (schedulingID == FIXED_INTERVAL) {
            if (interval * numOrigs > range) { // if needed modifies interval to make time for all txs
                interval = range / numOrigs;
            }
            for (int i = 0; i < numOrigs; i++) {
                steps[i] = startStep + interval * i;
            }
        } else if (schedulingID == RANDOM_INTERVAL) {
            interval = generateStartStep(range / numOrigs);
            for (int i = 0; i < numOrigs; i++) {
                steps[i] = startStep + interval * i;
            }
        } else if (schedulingID == UNORDERED) {
            steps[0] = generateStartStep(range) + start;
            for (int i = 1; i < numOrigs; i++) {
                steps[i] = generateStartStep(range - (int) steps[i-1]) + steps[i-1];
            }
        } else if (schedulingID == SIMULTANEOUS || range <2) {
            long step = generateStartStep(range) + start;
            Arrays.fill(steps, step);
        }
    }

    @Override
    public String getModelName() {
        return "FanIn";
    }

    private boolean isValidStep(long step) {
        return (step - startStep) % interval == 0;
    }

    @Override
    public void sendTransactions(long step, Account account) {
        for (int i = 0; i < origList.size(); i++) {
            if (steps[i] == step) {
                Account orig = origList.get(i);
                this.transactionAmount = new TargetedTransactionAmount(orig.getBalance(), this.random, true);
                makeTransaction(step, this.transactionAmount.doubleValue(), orig, account, AbstractTransactionModel.NORMAL_FAN_IN);
            }
        }
    }
}
