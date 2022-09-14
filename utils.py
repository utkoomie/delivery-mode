#------------------------------------------------------------------
# Support utilities for machine-learning analysis using CDC data
# 
# Copyright 2018-2022 Karl W. Schulz
# 
# Dell Medical School, University of Texas
#------------------------------------------------------------------

import os.path
import csv
import logging
import pandas
import numpy
import sys
import time
import matplotlib.pyplot as plt
import configparser
import textwrap 
import multiprocessing

#---
# Initialize logger
def initLogger(args):
    logging.basicConfig(format="%(message)s",level=logging.INFO,stream=sys.stdout)
    if args is None:
        return

    # allow user-provided loglevel (e.g. --log=DEBUG or --log=debug)
    valid_logLevels = ["DEBUG","INFO","WARNING","ERROR","CRITICAL"]
    if args.loglevel:
        args.loglevel=args.loglevel.upper()
        if args.loglevel not in valid_logLevels:
            ERROR("Invalid log level specified: %s" % args.loglevel)
        else:
            logger = logging.getLogger()
            logger.setLevel(args.loglevel)
            logging.debug("Enabled user-supplied log-level -> %s" % args.loglevel)

#---
# helper class to parse runtime configuration options
class runtimeConfig(object):
    def __init__(self,args):

        logging.info("\n--")
        logging.info("Parsing runtime options...")

        if args.configFile:
            self.ConfigFile=args.configFile
        else:
            self.ConfigFile="config.ini"
            
        logging.info("Using config from file = %s" % self.ConfigFile)
        if os.path.isfile(self.ConfigFile):
            self.options = configparser.ConfigParser(inline_comment_prefixes='#',interpolation=configparser.ExtendedInterpolation())
        else:
            ERROR("\nError: unable to access runtime configuration input file -> %s" % self.ConfigFile)
        
        try:
            self.options.read(self.ConfigFile)
        except Exception as e:
            logging.error("\n--ERROR" + "-" * 10 + "\n")
            logging.error(e)
            ERROR("Unable to parse runtime config file: %s" % self.ConfigFile)

        self.Years            = [year.strip() for year in self.options.get('cdc/config','years').split(',')]
        self.denomVars        = [var.strip() for var in self.options.get('cdc/config','denomVars').split(',')]
        self.denomVarsRevised = [var.strip() for var in self.options.get('cdc/config','denomVarsRevised').split(',')]

        self.numRecordsPerYear = self.options.getint('cdc/config','numRecordsPerYear')

        if args.year:
            self.Years = [args.year]
            logging.info("Overriding analysis year using command-line option: %s" % args.year)

        logging.info("   --> years = %s" % self.Years)
        logging.info("   --> numRecordsPerYear = %i " % self.numRecordsPerYear)

        # input sanity checks
        self.Years = list(map(int,self.Years))
        for year in self.Years:
            logging.info("")
            if year > 2017 or year < 2003:
                ERROR("Invalid year specified (%s): expecting range between 2003-2017" % year)

            numFile = self.options.get('cdc-raw/' + str(year),'numeratorFile')
            if os.path.isfile(numFile):
                logging.info("   --> %s/numeratorFile = %s" % (year,numFile))
            else:
                ERROR("Error: unable to access cdc numerator file -> %s" % numFile)

            denomFile = self.options.get('cdc-raw/' + str(year),'denomFile')
            if os.path.isfile(numFile):
                logging.info("   --> %s/denomFile     = %s" % (year,denomFile))
            else:
                ERROR("Error: unable to access cdc denominator file -> %s" % denomFile)

#---
# Simple error wrapper to include exit
def ERROR(output):
    logging.error(output)
    sys.exit()

#---
# Add provided labels to existing matplotlib barchart
def addBarChartLabels(bar,labels,offsets=None,percentage=False,fixedOffset=0):


    count = 0

    for rect in bar:
        if offsets is None:
            height=rect.get_height()
        else:
            height=rect.get_height() + offsets[count]

        textyloc = height - fixedOffset
        if percentage:
            plt.text(rect.get_x() + rect.get_width()/2.0, textyloc, '%.0f%%' % float(labels[count]), ha='center',
                     va='bottom',size='small',weight='light',fontsize=7.5)
        else:
            plt.text(rect.get_x() + rect.get_width()/2.0, textyloc, format(labels[count],','), ha='center',
                     va='bottom',size='small',weight='light',fontsize=7)
        count += 1

    # Add thousands separator in ylabels
    plt.gca().get_yaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

#---
# Utility routine to document datapoints dropped during filtering
def pdCount(data,comment=None,start=None):
    if comment is None:
        print("# of datapoints remaining {:42} = {:10,}" .format(' ',data.shape[0]))
    else:
        if start is None:
            print("# of datapoints remaining {:42} = {:10,}" .format(comment,data.shape[0]))
        else:
            dropped=start-data.shape[0]
            percentage = 100.0*(start - data.shape[0])/start
            print("# of datapoints remaining {:42} = {:10,} (dropped {:10,} : {:4.1f}%)"
                  . format(comment,data.shape[0],dropped,percentage))



#---
# Simple timer class to measure wall-clock execution time between
# start/stop boundaries
class timer(object):
    def __init__(self):
        self.t0 = 0

    def start(self):
        self.t0 = time.time()

    def stop(self,label=None,logFile=None):
        self.t1 = time.time()
        if label is not None:
            print("Total time %s = %.3f (secs)" % (label,(self.t1-self.t0)) )
            if logFile is not None:
                logFile.write("Total time %s = %.3f (secs)\n" % (label,(self.t1-self.t0)) )
        else:
            print("Total time = %.3f (secs)" % (self.t1-self.t0) )
            if logFile is not None:
                logFile.write("Total time = %.3f (secs)\n" % (self.t1-self.t0) )
        
