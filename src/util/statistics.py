import matplotlib.pyplot as plt


def plot_app_consumption(agregado, app):
    for i in range(app.shape[1]):
        plt.subplots(figsize=(16, 10))
        plt.title('Consumo Elétrico')
        plt.plot(agregado, label='Agregado')
        plt.plot(app[:, i:i + 1], label='Appliance {}'.format(i + 1))
        plt.legend()
        plt.show()


def plot_multi_app_consumption(app1, app2, path):
    plt.subplots(figsize=(16, 10))
    plt.subplot(211)
    plt.title('Consumo Elétrico - Real')
    plt.ylim(-0.05, 1)
    plt.ylabel('Proporção')
    plt.xlabel('Amostras')
    for i in range(app1.shape[1]):
        plt.plot(app1[:, i:i + 1], label='Appliance {}'.format(i + 1))
    plt.legend()

    plt.subplot(212)
    plt.title('Consumo Elétrico - Predição')
    plt.ylim(-0.05, 1)
    plt.ylabel('Proporção')
    plt.xlabel('Amostras')
    for i in range(app2.shape[1]):
        plt.plot(app2[:, i:i + 1], label='Appliance {}'.format(i + 1))
    plt.legend()

    plt.savefig(path, format='png', dpi=450)
    plt.clf()
    plt.close()


def plot_multi_app_consumption_individual(app1, app2, path):
    for i in range(app1.shape[1]):
        plt.subplots(figsize=(16, 10))
        plt.subplot(211)
        plt.title('Consumo Elétrico - Real')
        plt.ylim(-0.05, 1)
        plt.ylabel('Proporção')
        plt.xlabel('Amostras')
        plt.plot(app1[:, i:(i + 1)], label='Appliance {}'.format(i + 1))
        plt.legend()

        plt.subplot(212)
        plt.title('Consumo Elétrico - Predição')
        plt.ylim(-0.05, 1)
        plt.ylabel('Proporção')
        plt.xlabel('Amostras')
        plt.plot(app2[:, i:(i + 1)], label='Appliance {}'.format(i + 1))
        plt.legend()

        plt.savefig(path + '_{}'.format(i + 1), format='png', dpi=450)
        plt.clf()
        plt.close()
