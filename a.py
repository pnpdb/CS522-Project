import pandas as pd
import matplotlib.pyplot as plt

def plot(xValue, yValue, lines, title="Plot", xLabel=None, yLabel=None, colors = None, df=None, subtitle=None, ymin = None, ymax = None):
    if colors is None:
        colors = ['green', 'red', 'blue', 'brown', 'pink', 'black']
    if xLabel is None:
        xLabel = xValue.replace("x['","")
        xLabel = xLabel.replace("']","")
    if yLabel is None:
        yLabel = yValue.replace("y['","")
        yLabel = yLabel.replace("']","")

    fig, ax = plt.subplots()
    plt.title(title+("" if subtitle is None else ("\n"+subtitle)))
    i = 0
    for k, v in lines.items():
        index_line = df.apply(lambda df: eval(v), axis=1)
        df_line    = df[index_line]
        if(len(df_line) == 0):
            continue
        x = df_line.apply(lambda x: eval(xValue), axis=1)
        y = df_line.apply(lambda y: eval(yValue), axis=1)
        ax.plot(x, y, colors[i], label=k)
        ax.set_ylim(ymin = ymin, ymax = ymax)
        plt.scatter(x, y, marker="o", s=10)

        i = (i+1) % len(colors)

    plt.legend()
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

origin_train_set_sizes = [2000, 4000, 5000, 8000, 10000, 15000, 20000]
# origin_train_set_sizes = [5000, 10000, 15000, 20000]
noisy_train_set_sizes  = [(4000, 1000), (8000, 2000), (12000,3000), (15000, 5000)]

if __name__ == '__main__':
    df = pd.read_csv("saving/noise_effect_bert_0424.csv")
    print(df)

    for i in range(2):
        # Plot training set size vs. Macro F1

        # x coordinate
        if i == 0:
            xValue  = "x['Origin']+x['Noise']"
            xLabel  = "Training set total size\nnoisy sets: %s" % \
                      str([str(x[0])+'+'+str(x[1]) for x in noisy_train_set_sizes]).replace("\'","")
        else:
            xValue  = "x['Origin']"
            xLabel  = "Training set origin part size\nnoisy sets: %s" % \
                      str([str(x[0])+'+'+str(x[1]) for x in noisy_train_set_sizes]).replace("\'","")

        # y coordinate
        yValue  = "y['Macro F1']"

        # Divide experiments into several groups, each will be plotted as a line
        len1 = len(origin_train_set_sizes)
        len2 = len(noisy_train_set_sizes)
        lines = { # each item: name, filter
            'Original Data':       "int((df['Experiment']-1)/%d)==0"%len1,
            'Mislabeled Noise':    "int((df['Experiment']-1-%d)/%d)==0 and df['Experiment']-1-%d>=0"%(len1,len2,len1),
            'Irrelevant Noise':    "int((df['Experiment']-1-%d)/%d)==1"%(len1,len2),
            'Translated Noise(0% mislabeled)':    "int((df['Experiment']-1-%d)/%d)==2"%(len1,len2),
            'Translated Noise(50% mislabeled)':   "int((df['Experiment']-1-%d)/%d)==3"%(len1,len2),
            'Translated Noise(100% mislabeled)':  "int((df['Experiment']-1-%d)/%d)==4"%(len1,len2),
        }

        plot(xValue = xValue, yValue = yValue, lines = lines,
                    xLabel = xLabel, title = "BERT effected by various noises", ymin = 0.55, ymax = 0.79, df=df)

