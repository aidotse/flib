package amlsim.model.normal;

import amlsim.AMLSim;
import amlsim.Account;
import amlsim.AccountGroup;
import amlsim.TargetedTransactionAmount;
import amlsim.model.AbstractTransactionModel;

import java.util.List;
import java.util.Random;

/**
 * Send money only for once to one of the neighboring accounts regardless the
 * transaction interval parameter
 */
public class SingleTransactionModel extends AbstractTransactionModel {
    /**
     * Simulation step when this transaction is done
     */
    private long startStep = -1;
    private long endStep = -1;
    private long txStep = -1;
    
    private Random random;

    public SingleTransactionModel(
            AccountGroup accountGroup,
            Random random) {
        this.random = random;
        this.accountGroup = accountGroup;
        this.startStep = accountGroup.getStartStep();
        this.endStep = accountGroup.getEndStep();
        this.scheduleID = accountGroup.getScheduleID();
        this.interval = accountGroup.getInterval();
    }

    public String getModelName() {
        return "Single";
    }

    public void setParameters() {
        // The transaction step is determined randomly within the given range of steps
        this.txStep = this.startStep + this.random.nextInt((int) (endStep - startStep + 1));
    }

    public void sendTransactions(long step, Account account) {
        List<Account> beneList = account.getBeneList();
        int numBene = beneList.size();
        if (step != this.txStep || numBene == 0) {
            return;
        }

        TargetedTransactionAmount transactionAmount = new TargetedTransactionAmount(account.getBalance(), random,
                false);

        int index = this.random.nextInt(numBene);
        Account dest = beneList.get(index);
        this.makeTransaction(step, transactionAmount.doubleValue(), account, dest,
                AbstractTransactionModel.NORMAL_SINGLE);
    }
}
