import pandas as pd
import random

# note : after selecting data, it discards all other info and only keeps yield right now. May need to keep track of other info

# arm one: C-H arylation with substrate scope and lignad


class ChemArm:

    def __init__(self, val, name, url):
        self.val = val  # e.g. ('CC#N', 'CCC')
        self.name = name  # e.g. ('electrophile_smiles', 'nucleophile_smiles')
        self.data_url = url  # github url that contains raw data
        self.num_draw = 0  # how many times has this arm been played

        df = pd.read_csv(self.data_url)

        if type(self.name) == str:  # single element "tuple" becomes str
            if self.name not in df.columns:
                raise ValueError('name does not exist: ' + str(self.name))
            if self.val not in df[self.name].unique():
                raise ValueError('value does not exist: ' + str(self.val))
            df = df[[self.name, 'yield']]
        else:  # here name and val are passed in as tuple
            for i in range(len(self.name)):
                n = self.name[i]
                v = self.val[i]
                if n not in df.columns:
                    raise ValueError('name does not exist: ' + str(n))
                if v not in df[n].unique():
                    raise ValueError('value does not exist: ' + str(v))
            df[name] = df[list(name)].apply(tuple, axis=1)
            df = df[[name, 'yield']]

        self.data = df.loc[df[self.name] == self.val]['yield'].tolist()
        self.data = [d/100 for d in self.data]  # scale yield
        self.data_copy = self.data.copy()  # since pop self.data, need a copy to reset for each simulation

        return

    def draw(self):  # shuffle the data and pop
        random.shuffle(self.data)
        self.num_draw = self.num_draw + 1
        return self.data.pop()

    def reset(self):  # reset data between simulations
        self.data = self.data_copy.copy()


class ChemArmBinary(ChemArm):

    def __init__(self, val, name, url, cutoff):
        super().__init__(val, name, url)
        self.cutoff = cutoff
        self.data_binary = [int(d > cutoff) for d in self.data]




if __name__ == '__main__':

    val = ('Cy-BippyPhos')
    name = ('ligand_name')
    url = 'https://raw.githubusercontent.com/beef-broccoli/ochem-data/main/deebo/aryl-scope-ligand.csv'

    c = ChemArmBinary(val, name, url, 0.2)
    print(c.data)
    print(c.data_binary)