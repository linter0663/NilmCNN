import matplotlib.pyplot as plt


def plot_consumption(agregado, app):
    plt.subplots(figsize=(16, 10))
    plt.title('Consumo Elétrico')
    plt.plot(agregado, label='Agregado')
    plt.plot(app, label='Appliance')
    plt.legend()
    plt.show()


def plot_multi_app_consumption(app1, app2, path):
    plt.subplots(figsize=(16, 10))
    plt.subplot(211)
    plt.title('Consumo Elétrico - Real')
    plt.ylim(-0.05, 1)
    plt.ylabel('Proporção')
    plt.xlabel('Amostras')
    plt.plot(app1, label='Appliance')
    plt.legend()

    plt.subplot(212)
    plt.title('Consumo Elétrico - Predição')
    plt.ylim(-0.05, 1)
    plt.ylabel('Proporção')
    plt.xlabel('Amostras')
    plt.plot(app2, label='Appliance')
    plt.legend()

    plt.savefig(path, format='png', dpi=450)
    plt.clf()
    plt.close()
