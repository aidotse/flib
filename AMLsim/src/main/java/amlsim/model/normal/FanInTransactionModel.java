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

        this.startStep = accountGroup.getStartStep();
        this.endStep = accountGroup.getEndStep();
        this.scheduleID = accountGroup.getScheduleID();
        this.interval = accountGroup.getInterval();

        this.random = random;
    }

    public void setParameters() {

        // Set members
        List<Account> members = accountGroup.getMembers();
        Account mainAccount = accountGroup.getMainAccount();
        bene = mainAccount != null ? mainAccount : members.get(0); // The main account is the beneficiary
        for (Account orig : members) { // The rest of accounts are originators
            if (orig != bene)
                origList.add(orig);
        }

        // Set transaction schedule
        int numOrigs = origList.size();

        steps = new long[numOrigs];

        int range = (int) (this.endStep - this.startStep + 1);// get the range of steps

        if (scheduleID == FIXED_INTERVAL) {
            if (interval * numOrigs > range) { // if needed modifies interval to make time for all txs
                interval = range / numOrigs;
            }
            for (int i = 0; i < numOrigs; i++) {
                steps[i] = startStep + interval * i;
            }
        } else if (scheduleID == RANDOM_INTERVAL) {
            interval = generateFromInterval(range / numOrigs) + 1;
            for (int i = 0; i < numOrigs; i++) {
                steps[i] = startStep + interval * i;
            }
        } else if (scheduleID == UNORDERED) {
            steps[0] = generateFromInterval(range) + this.startStep;
            for (int i = 1; i < numOrigs; i++) {
                steps[i] = generateFromInterval(range - (int) steps[i - 1]) + steps[i - 1];
            }
        } else if (scheduleID == SIMULTANEOUS || range < 2) {
            long step = generateFromInterval(range) + this.startStep;
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
                makeTransaction(step, this.transactionAmount.doubleValue(), orig, account,
                        AbstractTransactionModel.NORMAL_FAN_IN);
            }
        }
    }
}
