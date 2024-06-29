import tkinter as tk

# Functions, which define the required prompts.

class SliceEntry:
    def __init__(self, root):
        self.entry_var = None

        root.geometry('300x100')
        self.L1 = tk.Label(root, text='Enter slices').grid(row=0)
        self.E1 = tk.Entry(root)
        self.E1.grid(row=0, column=1)
        tk.Button(root, text='OK', command=self.save_entry).grid(row=2, column=0, sticky=tk.W, pady=4)

    def save_entry(self):
        self.entry_var = self.E1.get()

class B1MappingOptionsUpdate:
    def __init__(self, root, optsdef):
        self.opts       = optsdef
        self.root       = root
        self.updated    = []
        self.fields     = {'Enter acquisition dimension:': self.opts['DIMB1'], 'Enter default WHICHSLICES:': self.opts['WHICHSLICES'],
                           'Enter default FIGWIDTH:': self.opts['FIGWIDTH'], 'Enter colormap name:': self.opts['COLORMAP'],
                           'Enter default RELPHASECHANNEL:': self.opts['RELPHASECHANNEL'], 'Enter USEMEAN:': self.opts['USEMEAN'],
                           'Enter SHOWMAPS:': self.opts['SHOWMAPS']}

        def callback(event):
            self.save_entry()

        E = self.make_form()
        T = tk.Text(self.root, height=2)
        T.pack()
        T.insert(tk.END, 'Press enter when done')
        self.root.bind('<Return>', callback)

    def make_form(self):
        self.entries = []
        for item in self.fields.items():
            row     = tk.Frame(self.root)
            label   = tk.Label(row, width=30, text=item[0])
            entry   = tk.Entry(row)
            entry.insert(10, item[1])
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            label.pack(side=tk.LEFT)
            entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
            self.entries.append((item[0], entry))

    def save_entry(self):
        for entry in self.entries:
            self.updated.append(entry[1].get())
        self.opts.update(zip(self.opts, self.updated))

        self.opts['DIMB1']              = int(self.opts['DIMB1'])
        self.opts['WHICHSLICES']        = int(self.opts['WHICHSLICES'])
        self.opts['FIGWIDTH']           = int(self.opts['FIGWIDTH'])
        self.opts['COLORMAP']           = str(self.opts['COLORMAP'])
        self.opts['RELPHASECHANNEL']    = int(self.opts['RELPHASECHANNEL'])
        self.opts['USEMEAN']            = bool(int(self.opts['USEMEAN']))
        self.opts['SHOWMAPS']           = bool(int(self.opts['SHOWMAPS']))

class B1ShimmingOptionsUpdate:
    def __init__(self, root, optsdef):
        self.opts       = optsdef
        self.root       = root
        self.updated    = []
        self.fields     = {'Enter LAMBDA:': self.opts['LAMBDA'], 'Enter MASKSTAYSAME:': self.opts['MASKSTAYSAME'],
                           'Enter NOOFSTARTPHASES:': self.opts['NOOFSTARTPHASES'], 'Enter MEANPOWER:': self.opts['MEANPOWER'],
                           'Enter SUMUPTOTHATTHRESHOLD:': self.opts['SUMUPTOTHATTHRESHOLD'], 'Enter EFFVALUEFORCE:': self.opts['EFFVALUEFORCE']}

        def callback(event):
            self.save_entry()

        E = self.make_form()
        T = tk.Text(self.root, height=2)
        T.pack()
        T.insert(tk.END, 'Press enter when done')
        self.root.bind('<Return>', callback)

    def make_form(self):
        self.entries = []
        for item in self.fields.items():
            row     = tk.Frame(self.root)
            label   = tk.Label(row, width=30, text=item[0])
            entry   = tk.Entry(row)
            entry.insert(10, item[1])
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            label.pack(side=tk.LEFT)
            entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
            self.entries.append((item[0], entry))

    def save_entry(self):
        for entry in self.entries:
            self.updated.append(entry[1].get())
        self.opts.update(zip(self.opts, self.updated))

        self.opts['LAMBDA']                 = float(self.opts['LAMBDA'])
        self.opts['MASKSTAYSAME']           = int(self.opts['MASKSTAYSAME'])
        self.opts['NOOFSTARTPHASES']        = int(self.opts['NOOFSTARTPHASES'])
        self.opts['MEANPOWER']              = float(self.opts['MEANPOWER'])
        self.opts['SUMUPTOTHATTHRESHOLD']   = float(self.opts['SUMUPTOTHATTHRESHOLD'])
        self.opts['EFFVALUEFORCE']          = float(self.opts['EFFVALUEFORCE'])

