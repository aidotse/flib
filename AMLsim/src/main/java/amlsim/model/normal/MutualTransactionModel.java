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
        lender = mainAccount != null ? mainAccount : members.get(0); // The main account is the beneficiary
        for (Account debtor : members) { // The rest of accounts are originators
            if (debtor != lender)
                debtorList.add(debtor);
        }
        debtor = debtorList.get(0);
        debt = 0.0;

        int schedulingID = this.accountGroup.getScheduleID();
        int range = (int) (end - start + 1);// get the range of steps
        if (schedulingID == FIXED_INTERVAL) {
            this.interval = interval;
        } else if (schedulingID == RANDOM_INTERVAL || schedulingID == UNORDERED) {
            this.interval = generateFromInterval(range) + (int) start;
        } else if (schedulingID == SIMULTANEOUS || range < 2) {
            this.interval = 1;
        }
    }

    @Override
    public String getModelName() {
        return "Mutual";
    }

    @Override
    public void sendTransactions(long step, Account account) {
        // Account debtor = account.getDebtor();
        // double debt = account.getDebt();
        // if (debtor != null && debt > 0) { // Return money from debtor to main account
        // if (debt > debtor.getBalance()) { // Return part of the debt
        // TargetedTransactionAmount transactionAmount = new
        // TargetedTransactionAmount(debtor.getBalance(), random, false);
        // double amount = transactionAmount.doubleValue();
        // makeTransaction(step, amount, debtor, account,
        // AbstractTransactionModel.NORMAL_MUTUAL);
        // account.setDebtor(debtor);
        // account.setDebt(debt - amount);
        // } else { // Return all the debt
        // makeTransaction(step, debt, debtor, account,
        // AbstractTransactionModel.NORMAL_MUTUAL);
        // account.setDebtor(null);
        // account.setDebt(0);
        // }
        // } else { // Lend money to a random neighbour
        // List<Account> origs = account.getOrigList();
        // if (origs.isEmpty()) {
        // return;
        // } else {
        // int i = random.nextInt(origs.size());
        // debtor = origs.get(i);
        // }
        // TargetedTransactionAmount transactionAmount = new
        // TargetedTransactionAmount(account.getBalance(), random, false);
        // if (!account.getBeneList().contains(debtor)) { // TODO: this effects the
        // structure of the graph, fix? This function exist cuz the python code only
        // creates mutal models with two accounts, in more complex networks this if is
        // needed
        // account.addBeneAcct(debtor); // Add a new destination
        // }
        // debt = transactionAmount.doubleValue();
        // makeTransaction(step, debt, account, debtor,
        // AbstractTransactionModel.NORMAL_MUTUAL);
        // account.setDebtor(debtor);
        // account.setDebt(debt);
        // }

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
