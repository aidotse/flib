package amlsim.model.normal;

import amlsim.*;
import amlsim.model.AbstractTransactionModel;

import java.util.*;

/**
 * Return money to one of the previous senders
 */
public class MutualTransactionModel extends AbstractTransactionModel {

    private Account lender; // The destination (beneficiary) account
    private List<Account> debtorList = new ArrayList<>(); // The origin (originator) accounts
    private Account debtor;
    private double debt;
    private Random random;

    public MutualTransactionModel(
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
        lender = mainAccount != null ? mainAccount : members.get(0); // The main account is the beneficiary
        for (Account debtor : members) { // The rest of accounts are originators
            if (debtor != lender)
                debtorList.add(debtor);
        }
        debtor = debtorList.get(0);
        debt = 0.0;

        int range = (int) (this.endStep - this.startStep + 1);// get the range of steps
        if (scheduleID == FIXED_INTERVAL) {
            ;
        } else if (scheduleID == RANDOM_INTERVAL || scheduleID == UNORDERED) {
            this.interval = generateFromInterval(range) + (int) this.startStep;
        } else if (scheduleID == SIMULTANEOUS || range < 2) {
            this.interval = 1;
        }
    }

    @Override
    public String getModelName() {
        return "Mutual";
    }

    @Override
    public void sendTransactions(long step, Account account) {

        if (step == this.startStep) {
            int i = random.nextInt(debtorList.size());
            debtor = debtorList.get(i);
            TargetedTransactionAmount transactionAmount = new TargetedTransactionAmount(account.getBalance(), random,
                    false);
            debt = transactionAmount.doubleValue();
            makeTransaction(step, debt, account, debtor, AbstractTransactionModel.NORMAL_MUTUAL);
        } else if (step == this.startStep + interval) {
            if (debt > debtor.getBalance()) { // Return part of the debt
                TargetedTransactionAmount transactionAmount = new TargetedTransactionAmount(debtor.getBalance(), random,
                        false);
                double amount = transactionAmount.doubleValue();
                makeTransaction(step, amount, debtor, account, AbstractTransactionModel.NORMAL_MUTUAL);
                debt = debt - amount;
            } else { // Return all the debt
                makeTransaction(step, debt, debtor, account, AbstractTransactionModel.NORMAL_MUTUAL);
                debtor = null;
                debt = 0.0;
            }
        }
    }
}
