import matplotlib.pyplot as plt


rounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

fed_client_0 = [0.6786, 0.7232, 0.7402, 0.7517, 0.7639, 0.7709, 0.7778, 0.7794, 0.7801, 0.7799, 0.7801]
fed_client_1 = [0.6846, 0.7315, 0.7436, 0.7553, 0.7659, 0.7760, 0.7797, 0.7802, 0.7801, 0.7801, 0.7805]
fed_client_2 = [0.6858, 0.7348, 0.7442, 0.7462, 0.7663, 0.7714, 0.7772, 0.7793, 0.7799, 0.7805, 0.7799]

nofed_client_0 = [0.2419, 0.7660, 0.7758, 0.7762, 0.7765, 0.7765, 0.7765, 0.7767, 0.7761, 0.7764, 0.7765]
nofed_client_1 = [0.7583, 0.7583, 0.7584, 0.7583, 0.7576, 0.7577, 0.7577, 0.7580, 0.7583, 0.7583, 0.7581]
nofed_client_2 = [0.7583, 0.7584, 0.7586, 0.7584, 0.7583, 0.7581, 0.7582, 0.7586, 0.7581, 0.7581, 0.7583]
 

plt.plot(rounds, fed_client_0, color='C0')
plt.plot(rounds, fed_client_1, color='C0')
plt.plot(rounds, fed_client_2, color='C0')

plt.plot(rounds, nofed_client_0, color='C1')
plt.plot(rounds, nofed_client_1, color='C1')
plt.plot(rounds, nofed_client_2, color='C1')

plt.savefig('result.png')

