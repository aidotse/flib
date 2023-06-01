package amlsim.model.normal;

import amlsim.*;
import amlsim.model.AbstractTransactionModel;

import java.util.List;
import java.util.Random;

/**
 * Return money to one of the previous senders
 */
public class MutualTransactionModel extends AbstractTransactionModel {

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
    }

    @Override
    public String getModelName() {
        return "Mutual";
    }

    @Override
    public void sendTransactions(long step, Account account) {
        if ((step - this.startStep) % interval != 0)
            return;

        Account debtor = account.getDebtor();
        double debt = account.getDebt();
        if (debtor != null && debt > 0) { // Return money from debtor to main account
            if (debt > debtor.getBalance()) { // Return part of the debt
                TargetedTransactionAmount transactionAmount = new TargetedTransactionAmount(debtor.getBalance(), random,
                        false);
                double amount = transactionAmount.doubleValue();
                makeTransaction(step, amount, debtor, account, AbstractTransactionModel.NORMAL_MUTUAL);
                account.setDebtor(debtor);
                account.setDebt(debt - amount);
            } else { // Return all the debt
                makeTransaction(step, debt, debtor, account, AbstractTransactionModel.NORMAL_MUTUAL);
                account.setDebtor(null);
                account.setDebt(0);
            }
        } else { // Lend money to a random neighbour
            List<Account> origs = account.getOrigList();
            if (origs.isEmpty()) {
                return;
            } else {
                int i = random.nextInt(origs.size());
                debtor = origs.get(i);
            }
            TargetedTransactionAmount transactionAmount = new TargetedTransactionAmount(account.getBalance(), random,
                    false);
            if (!account.getBeneList().contains(debtor)) { // TODO: this effects the structure of the graph, fix? This
                                                           // function exist cuz the python code only creates mutal
                                                           // models with two accounts, in more complex networks this if
                                                           // is needed
                account.addBeneAcct(debtor); // Add a new destination
            }
            debt = transactionAmount.doubleValue();
            makeTransaction(step, debt, account, debtor, AbstractTransactionModel.NORMAL_MUTUAL);
            account.setDebtor(debtor);
            account.setDebt(debt);
        }
    }
}
