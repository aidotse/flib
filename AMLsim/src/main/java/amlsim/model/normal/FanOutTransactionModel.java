package amlsim.model.normal;

import amlsim.*;
import amlsim.model.AbstractTransactionModel;

import java.util.*;

/**
 * Distribute money to multiple neighboring accounts (fan-out)
 */
public class FanOutTransactionModel extends AbstractTransactionModel {

    // Originators and the main beneficiary
    private Account orig; // The destination (beneficiary) account
    private List<Account> beneList = new ArrayList<>(); // The origin (originator) accounts

    private long[] steps;

    private Random random;
    private TargetedTransactionAmount transactionAmount;

    public FanOutTransactionModel(
            AccountGroup accountGroup,
            Random random) {
        this.accountGroup = accountGroup;
        this.random = random;
    }

    public void setParameters(long start, long end, int interval) {
        super.setParameters(start, end, interval);
        if (this.startStep < 0) { // decentralize the first transaction step
            this.startStep = generateFromInterval(interval);
        }

        // Set members
        List<Account> members = accountGroup.getMembers();
        Account mainAccount = accountGroup.getMainAccount();
        orig = mainAccount != null ? mainAccount : members.get(0); // The main account is the beneficiary
        for (Account bene : members) { // The rest of accounts are originators
            if (bene != orig)
                beneList.add(bene);
        }

        // Set transaction schedule
        int schedulingID = this.accountGroup.getScheduleID();
        int numBenes = beneList.size();

        steps = new long[numBenes];

        int range = (int) (end - start + 1);// get the range of steps

        if (schedulingID == FIXED_INTERVAL) {
            if (interval * numBenes > range) { // if needed modifies interval to make time for all txs
                interval = range / numBenes;
            }
            for (int i = 0; i < numBenes; i++) {
                steps[i] = startStep + interval * i;
            }
        } else if (schedulingID == RANDOM_INTERVAL) {
            interval = generateFromInterval(range / numBenes);
            for (int i = 0; i < numBenes; i++) {
                steps[i] = startStep + interval * i;
            }
        } else if (schedulingID == UNORDERED) {
            steps[0] = generateFromInterval(range) + start;
            for (int i = 1; i < numBenes; i++) {
                steps[i] = generateFromInterval(range - (int) steps[i - 1]) + steps[i - 1];
            }
        } else if (schedulingID == SIMULTANEOUS || range < 2) {
            long step = generateFromInterval(range) + start;
            Arrays.fill(steps, step);
        }
    }

    @Override
    public String getModelName() {
        return "FanOut";
    }

    private boolean isValidStep(long step) {
        return (step - startStep) % interval == 0;
    }

    @Override
    public void sendTransactions(long step, Account account) {
        for (int i = 0; i < beneList.size(); i++) {
            if (steps[i] == step) {
                Account bene = beneList.get(i);
                this.transactionAmount = new TargetedTransactionAmount(bene.getBalance(), this.random, true);
                makeTransaction(step, this.transactionAmount.doubleValue(), bene, account,
                        AbstractTransactionModel.NORMAL_FAN_OUT);
            }
        }
    }
}
