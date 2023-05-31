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

    private Random random;

    private long[] steps;

    public ForwardTransactionModel(
            AccountGroup accountGroup,
            Random random) {
        this.accountGroup = accountGroup;
        this.random = random;
    }

    public void setParameters(int interval, long start, long end) {
        this.initialMainAccountID = accountGroup.getMainAccount().getID();
        super.setParameters(interval, start, end);

        // this will cause the forward transactions to start in [0, interval]
        if (this.startStep < 0) { // decentralize the first transaction step
            this.startStep = generateStartStep(interval);
        }
        int schedulingID = this.accountGroup.getScheduleID();

        // Set members
        List<Account> members = accountGroup.getMembers(); // get all members in accountgroup
        Account mainAccount = accountGroup.getMainAccount(); // get main account
        mainAccount = mainAccount != null ? mainAccount : members.get(0); // get main account (if not set, pick the
                                                                          // first member)

        // Set transaction schedule
        steps = new long[2]; // keep track of when the two first members should perform an action

        int range = (int) (end - start + 1);// get the range of steps

        // if simultaneous or model is only alive for one step, make them consecutive
        if (schedulingID == SIMULTANEOUS || range < 2) {
            long step = generateStartStep(range) + start; // generate a step in [start, end] randomly
            steps[0] = step;
            steps[1] = step + 1;
        } else if (schedulingID == FIXED_INTERVAL) { // if fixed interval, set steps to be evenly spaced
            for (int i = 0; i < 2; i++) {
                steps[i] = startStep + interval * i;
            }
        } else if (schedulingID == RANDOM_INTERVAL || schedulingID == UNORDERED) { // if unordered, set steps random
            for (int i = 0; i < 2; i++) {
                steps[i] = generateStartStep(range) + start;
            }
            Arrays.sort(steps); // make sure the steps are in order
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

        TargetedTransactionAmount transactionAmount = new TargetedTransactionAmount(account.getBalance(), random,
                false);

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
            this.makeTransaction(step, transactionAmount.doubleValue(), account, dest,
                    AbstractTransactionModel.NORMAL_FORWARD);
            this.accountGroup.setMainAccount(dest); // set the main account to be the destination
            index = (index + 1) % 2; // use the next time step for the next transaction

            // if we have done two transactions, reset the main account to initial account
            if (index == 0) {
                this.resetMainAccount();
            }
        }
    }
}
