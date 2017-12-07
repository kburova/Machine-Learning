from backprop import Data
from backprop import BackProp


def main():
    outfile = open('results.txt', 'w')
    d = Data('spambase.data')

    layer = [15]
    epochs = 500
    lrate = [0.25, 0.4, 0.65, 0.85, 1.0]

    for k in [10, 15, 30]:
        d.reduce(k)
        d.split(d.Z)
        print('pca = ', k)
        for i, lr in enumerate(lrate):
            n = k * len(lrate) + i
            print('Problem %d\n' % n, file=outfile)
            try:
                BackProp(d, lr, epochs, len(layer), layer).run(n, len(lrate), outfile)
            except OverflowError:
                print('Overflow')

    outfile.close()


main()