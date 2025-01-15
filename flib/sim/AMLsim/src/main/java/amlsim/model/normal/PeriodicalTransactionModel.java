package amlsim.model.normal;

import java.util.Random;

import amlsim.Account;
import amlsim.AccountGroup;
import amlsim.TargetedTransactionAmount;
import amlsim.model.AbstractTransactionModel;

import java.util.*;

/**
 * Send money to neighbors periodically
 */
public class PeriodicalTransactionModel extends AbstractTransactionModel {

    private int index = 0;
    private long startStep = -1;
    private long endStep = -1;
    private int scheduleID = -1;
    private int interval = -1;

    private Random random;

    private long steps = -1;

    private TargetedTransactionAmount targedTransactionAmount = null;
    private double amount = -1;

    public PeriodicalTransactionModel(
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
        int range = (int) (this.endStep - this.startStep + 1);// get the range of steps

        if (this.interval > range || this.interval < 0) {
            this.interval = range;
        }

        this.steps = this.startStep;

        switch (this.scheduleID) {
            case SIMULTANEOUS:
                break;
            case FIXED_INTERVAL:
                break;
            case RANDOM_INTERVAL:
                this.interval = generateFromInterval(range) + 1; // generate random period
                break;
            case UNORDERED:
                this.interval = generateFromInterval(range) + 1; // TODO: isValid breaks if interval is zero, fix this?
                break;
            default:
                this.steps = this.startStep;
                break;
        }
    }

    @Override
    public String getModelName() {
        return "Periodical";
    }

    private boolean isValidStep(long step) {
        boolean outside_range = step < this.startStep || step > this.endStep;

        if (outside_range) {
            return !outside_range;
        } else {
            boolean period_passed = (step - this.startStep) % this.interval == 0;
            return period_passed;
        }
    }

    @Override
    public void sendTransactions(long step, Account account) {

        if (!isValidStep(step) || account.getBeneList().isEmpty()) {
            return;
        }

        if (this.targedTransactionAmount == null) {
            this.targedTransactionAmount = new TargetedTransactionAmount(account.getBalance(), random, false);
            amount = this.targedTransactionAmount.doubleValue();
        }

        List<Account> dests = account.getBeneList();
        List<Account> members = accountGroup.getMembers();
        Set<Account> destsSet = new HashSet<>(dests);
        destsSet.retainAll(members);
        Account dest = destsSet.iterator().next();

        if (account.getBalance() < amount) {
            amount = 0.9 * account.getBalance();
        }
        this.makeTransaction(step, amount, account, dest, AbstractTransactionModel.NORMAL_PERIODICAL);

    }
}
