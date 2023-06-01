package amlsim.model.normal;

import amlsim.Account;
import amlsim.AccountGroup;
import amlsim.TargetedTransactionAmount;
import amlsim.model.AbstractTransactionModel;

import java.util.*;

/**
 * Send money received from an account to another account in a similar way
 */
public class ForwardTransactionModel extends AbstractTransactionModel {
    private int index = 0;
    private String initialMainAccountID;
    private long startStep = -1;
    private long endStep = -1;
    private int scheduleID = -1;
    private int interval = -1;

    private TargetedTransactionAmount targedTransactionAmount = null;
    private double amount = -1;

    private Random random;

    private long[] steps;

    public ForwardTransactionModel(
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
        this.initialMainAccountID = accountGroup.getMainAccount().getID();

        // this will cause the forward transactions to start in [0, interval]
        if (this.startStep < 0) { // decentralize the first transaction step
            this.startStep = generateFromInterval(interval);
        }

        // Set members
        List<Account> members = accountGroup.getMembers(); // get all members in accountgroup
        Account mainAccount = accountGroup.getMainAccount(); // get main account
        mainAccount = mainAccount != null ? mainAccount : members.get(0); // get main account (if not set, pick the
                                                                          // first member)
        // Set transaction schedule
        steps = new long[2]; // keep track of when the two first members should perform an action
        int range = (int) (this.endStep - this.startStep + 1);// get the range of steps

        // if simultaneous or model is only alive for one step, make them consecutive
        switch (this.scheduleID) {
            case SIMULTANEOUS:
                long step = generateFromInterval(range, (int) this.startStep); // generate a step in [start, end]
                steps[0] = step;
                steps[1] = step + 1;
                break;
            case FIXED_INTERVAL:
                if (interval > range) {
                    interval = range;
                }
                for (int i = 0; i < 2; i++) {
                    steps[i] = startStep + interval * i;
                }
                break;
            case RANDOM_INTERVAL:
                int random_interval = generateFromInterval(range);
                for (int i = 0; i < 2; i++) {
                    steps[i] = startStep + random_interval * i;
                }
                break;
            case UNORDERED:
                for (int i = 0; i < 2; i++) {
                    steps[i] = generateFromInterval(range, (int) this.startStep);
                }
                Arrays.sort(steps); // make sure the steps are in order
                break;
            default:
                steps[0] = this.startStep;
                steps[1] = this.startStep + 1;
                break;
        }
    }

    @Override
    public String getModelName() {
        return "Forward";
    }

    private void resetMainAccount() {
        List<Account> members = accountGroup.getMembers();
        for (int i = 0; i < members.size(); i++) { // go through the members to find initial main account
            if (members.get(i).getID().equals(this.initialMainAccountID)) {
                Account nextMainAccount = members.get(i); // get account
                this.accountGroup.setMainAccount(nextMainAccount); // set account as main account
                break;
            }
        }
    }

    @Override
    public void sendTransactions(long step, Account account) {

        if (this.targedTransactionAmount == null) {
            this.targedTransactionAmount = new TargetedTransactionAmount(account.getBalance(), random,
                    false);
            amount = this.targedTransactionAmount.doubleValue();
        }

        // get the next destination account by looking at the intersection between
        // beneficiary list and account group members
        List<Account> dests = account.getBeneList();
        List<Account> members = accountGroup.getMembers();
        Set<Account> destsSet = new HashSet<>(dests);
        destsSet.retainAll(members); // get overlap between beneficiaries and account group
                                     // members

        int numDests = destsSet.size();
        if (numDests == 0) {
            return;
        }

        // make transactions if step is correct
        if (steps[index] == step) {
            Account dest = destsSet.iterator().next(); // it is unlikely that this set is larger than 1
            if (account.getBalance() < amount) {
                amount = 0.9 * account.getBalance();
            }
            this.makeTransaction(step, amount, account, dest,
                    AbstractTransactionModel.NORMAL_FORWARD);
            this.accountGroup.setMainAccount(dest); // set the main account to be the destination
            index = (index + 1) % 2; // get index of next time for action

            // if we have done two transactions, reset the main account to initial account
            if (index == 0) {
                this.resetMainAccount();
            }
        }
    }
}
