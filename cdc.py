# -------------------------------------------------------------------
# Support class to read and parse raw CDC data.
# CDC files publicly available at: 
#   https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm#Births. 
#
# Copyright 2018-2022 Karl W. Schulz
#
# Dell Medical School, University of Texas
# -------------------------------------------------------------------

import os.path
import logging
import utils
import sys, traceback
import configparser
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm

# A general comment on python string indexing which is used in a
# variety of parsing routines in this file. First off, the indexing is
# 0-based -- nothing wrong with that, but the rub is that a string
# slice does not include the last position. So, if you want to read
# from fixed column positions (10-14) from a CDC file, the relevant
# python indexing is [9-14].  I promise I didn't make this goofiness
# up.
# 
# https://www.pythoncentral.io/cutting-and-slicing-strings-in-python/


#---
# Define convenience aliases for logger(s)
info  = utils.logging.info
debug = utils.logging.debug
error = utils.logging.error
ERROR = utils.ERROR

#---
# CDC raw data reader helper class
class cdc(object):

    def __init__(self,args,config):

        logging.info("\n--")
        logging.info("Initializing raw reads from desired CDC files...")

        self._fetalAnomalyCounts = defaultdict(int)
        self._missingDataCounts  = defaultdict(int)
        self._currentYear        = None
        
        # variable field length and starting indices
        self.varLength    = defaultdict(int)
        # variable type (int,float,unknown for string)
        self.varType      = defaultdict(str)
        # array of variable locations on a per-year basis
        self.varIndex     = {}  
        # array of corresponding variable flag index on a per-year basis
        self.varIndexFlag = {}  

        # birth stats
        self.fetalDeaths_us              = defaultdict(int)
        self.births_us                   = defaultdict(int)
        self.infantDeaths_us             = defaultdict(int)
        self.infantDeathsU28_us          = defaultdict(int)
        self.infantDeaths_us_weighted    = defaultdict(float)
        self.infantDeathsU28_us_weighted = defaultdict(float)
        self.fetalMortality              = defaultdict(int)

        # plotting settings
        self.plotType            = '.pdf'
        self.titleSize           = 9
        self.ylabelSize          = 9
        self.xTicSize            = 8
        self.yTicSize            = 8

        # Initialize storage for variable reporting statistics on a per year basis
        self.varsReporting = {}
        self.varsTotal     = {}

        # Cache runtime options from config file
        self.enablePlots = config.options.getboolean('cdc/config','enablePlots', fallback=False)
        self.deactivateFD_GAUnder20 = config.options.getboolean('cdc/config','deactivateFD_GAUnder20', fallback=True)
        self.deactivateFD_territories = config.options.getboolean('cdc/config','deactivateFD_territories', fallback=True)
        self.deactivateFD_withoutValidAnomaly = config.options.getboolean('cdc/config','deactivateFD_withoutValidAnomaly', fallback=False)
        self.deactivateFD_withoutValidRiskFactors = config.options.getboolean('cdc/config','deactivateFD_withoutValidRiskFactors', fallback=False)
        self.periodData = config.options.getboolean('cdc/config','periodData', fallback=False)

        firstYear = True

        for year in config.Years:
            # cache beginning column indices for variables that change
            # location in file from year to year (thanks CDC)
            info("")
            self.cacheVarConfig(config,config.denomVars)
            info("")
            self.cacheVarConfig(config,config.denomVarsRevised)

    # Read raw data from denominator (birth) file. 
    # --
    def loadDenominatorFiles(self,db,config):
        numReadTotal=0
        numActiveTotal = 0
        numstate_differences = 0

        self.denomData = []

        info("")
        info("-"*60)
        info("Reading birth data from denominator file(s)...")

        for year in config.Years:
            self.numReadYearly = 0

            self._currentYear = year
            self.varsReporting[year] = defaultdict(int)
            self.varsTotal[year]     = defaultdict(int)

            try:
                File = config.options.get('cdc-raw/' + str(year),'denomFile')

                info("\nRaw data configuration (year=%i):" % year)

                with tqdm(total=os.path.getsize(File)) as progressBar:
                    file = open(File,'r')
                    for line in file:

                        progressBar.update(len(line))

                        numReadTotal  += 1
                        self.numReadYearly += 1

                        datum = self.parseDenominatorRecord(config,line)

                        if not datum['active']:
                            continue

                        self.denomData.append(datum)

                        if config.numRecordsPerYear != -1:
                            if numReadTotal >= config.numRecordsPerYear:
                                break
                    

            except Exception:
                print('-'*60)
                traceback.print_exc()
                print('-'*60)
                utils.ERROR("\nError:Unable to complete parse of raw contents from denominator file = %s" % File)

            info("\nTotal number of raw births read           = {:10,}".format(numReadTotal))
            info("--> # of U.S. births (50 states)          = {:10,}".format(self.births_us[year]))

            # convert to dataFrame
            dataFrame = pandas.DataFrame.from_dict(self.denomData)
            return(dataFrame)

    # --
    # Parse fields of raw CDC entry (denominator file format)
    def parseDenominatorRecord(self,config,line):

        datum = {}

        #  we start by assuming all records are active and prune based on additional tests
        datum['active'] = True

        # read all vars defined in denomVars runtime config
        for var in config.denomVars:

            # special logic for 2014 and newer: the 'revision'
            # variable is no longer present beginning in 2014; we set
            # it directly to 'A' to assume the revised certificate as
            # a result.

            myYear = int(self._currentYear)
            if myYear >= 2014 and var == 'revision':
                datum[var] = 'A'
            else:
                datum[var] = self.readVar(var,line,self.varType[var])

        # only keep US residents
        assert(datum['restatus'] >= 1 and datum['restatus'] <= 4)
        if datum['restatus'] == 4:
            self.deactivateRecord(datum,'BirthForeignResident')
        else:
            self.births_us[self._currentYear] += 1

        # read all remaining desired vars defined in denomVarsRevised runtime config
        if datum['revision'] == 'A':
            for var in config.denomVarsRevised:
                if (self.varIndex[var][self._currentYear] == 'skip'):
                    datum[var] = 'skip'
                else:
                    datum[var] = self.readVar(var,line,self.varType[var])

        return(datum)


    def deactivateRecord(self,datum,comment):
        debug("Deactivating record -> %s" % comment)

        datum['active'] = False
        if 'inactiveReason' in datum:
            datum['inactiveReason'] += "," + comment
        else:
            datum['inactiveReason'] = comment

    # --
    # query starting variable record location and field length
    def queryVarLocation(self,varname,year):
        if varname not in self.varLength:
            utils.ERROR("\nError: %s variable not cached in varLength" % varname)
        if varname not in self.varIndex:
            utils.ERROR("\nError: %s variable not cached in varIndex" % varname)
        if year not in self.varIndex[varname]:
            utils.ERROR("\nError: %s variable not cached in for year = %s" % (varname,year))
        return self.varLength[varname],self.varIndex[varname][year],self.varIndexFlag[varname][year]

    # --
    # load desired variable stored at "index" if the corresponding
    # reporting flag stored at "index + offset" has a value of 1
    def storeIfReported(self,datum,line,variableName,index,offset):
        self.varsTotal[self._currentYear][variableName] += 1
        flagIndex = index + offset
        debug("(index,flag) index for %s = (%i,%i)" % (variableName,index,flagIndex))
        if line[flagIndex] == '1':
            datum[variableName] = line[index]
            self.varsReporting[self._currentYear][variableName] += 1

    def summarizeVarReporting(self,Years):
        info("\n--")
        
        for year in Years:
            info("Year: %s" % year)
            print("len of varsTotal = %i" % len(self.varsTotal[year]))
            print(self.varsTotal[year])
            print(" ")
            print("len of varsReporting = %i" % len(self.varsReporting[year]))
            print(self.varsReporting[year])

            for key in self.varsTotal[year]:
                per = 100 * self.varsReporting[year][key] / self.varsTotal[year][key]
                info("  %-12s reported %9i records of %9i total (%4.1f %%)" % (key,self.varsReporting[year][key],self.varsTotal[year][key],per))

    #----------------------------------------------------------
    # Read variable from field and convert to desired dataType
    # --> column location/length/reporting flag for a given 
    # --> variable come from config file parsing
    # ---------------------------------------------------------

    def readVar(self,varName,line,dataType):
        recordlen,index,flag = self.queryVarLocation(varName,self._currentYear)

        # Step 1: check if reporting flag is used and active for this
        # variable. If flag is empty (spaces), we assume the variable
        # is not reported.

        notReported = -999

        if flag != -999:
            if line[flag-1].isspace():
                return notReported
            try:
                value = (int)(line[flag-1])
            except:
                record = self.numReadYearly
                info("Unable to read reporting flag for variable %s" % (varName))
                info("--> record # = %i" % record)

            # expected values are:
            # 0 -> item reported in neither the current or previous year
            # 1 -> item reported in both current and previous year
            # 2 -> item reported in the previous but not in current year
            # 3 -> item reported in in current, but not previous year

            if value == 0:
                debug("%s: reporting flag value of 0 found" % varName)
                return notReported
            elif value == 2:
                return notReported
            elif value == 3:
                debug("reporting flag value of 3 found")
            elif value == 1:
                debug("reporting flag value of 1 found")
            else:
                ERROR("Unknown reporting flag read -> %i" % value)

        # Step 2: read variable data
        try:
            if dataType == 'float':
                value  = (float)(line[index-1:index-1+recordlen])
            elif dataType == 'int':
                entry = line[index-1:index-1+recordlen]
                if entry.isspace():
                    value = -999
                else:
                    value  = (int)(entry)
            elif dataType == 'unknown':
                value  = (line[index-1:index-1+recordlen])
            else:
                ERROR("unknown dataType (%s) in readVar" % dataType)

        except:
            record = self.numReadYearly
            value  = line[index-1:index-1+recordlen]
            info("Unable to parse variable %s, expecting %s datatype" % (varName,dataType))
            info("--> record # = %i" % record)
            if not value:
                ERROR("--> string read is empty")
            else:
                ERROR("--> string read is (%s)" % value)

        return value

    #---
    # Cache information related to variable location in CDC files
    # (variable length/type, starting column, flag information). 
    # Locations can change from year to year (thanks CDC).
    def cacheVarConfig(self,config,varList):
        firstYear = True

        for year in config.Years:

            for var in varList:
                if firstYear:
                    try:
                        self.varLength[var] = config.options.getint('cdc/varindex/'+ var,'len')
                    except configparser.Error:
                        utils.ERROR("\nError: Unable to read value for cdc/varindex/%s/len" % var)

                    assert(self.varLength[var] > 0)

                        
                    try:
                        self.varType[var] = config.options.get('cdc/varindex/'+ var,'type',fallback=None)
                    except configparser.Error:
                        utils.ERROR("\nError: Unable to read value for cdc/varindex/%s/type" % var)

                    if self.varType[var] is None:
                        utils.ERROR("\nError: variable type not provided for var=%s" % var)
                            
                    # storage for indices on a per year basis
                    self.varIndex[var]     = defaultdict(int)
                    self.varIndexFlag[var] = defaultdict(int)
                            
                try:
                    value = config.options.get('cdc/varindex/'+ var,str(year))
                except configparser.Error:
                    utils.ERROR("\nError: Unable to read varindex for var=%s (year = %i)" % (var,year))
                                
                if not value:
                    utils.ERROR("\nError: varindex for var=%s (year = %i) is empty" % (var,year))

                if value != "skip": 
                    # convert to int
                    try:
                        self.varIndex[var][year] = int(value)
                    except ValueError:
                        utils.ERROR("\nError: Unable to convert varindex for var=%s (year = %i) to int" % (var,year))
                
                        assert(self.varIndex[var][year] > 0)
                else:
                    self.varIndex[var][year] = value

                # check if flag index exists for this var
                if config.options.has_option('cdc/varindex/' + var,str(year)+'_flag'):
                    self.varIndexFlag[var][year] = config.options.getint('cdc/varindex/' + var,str(year)+'_flag')
                else:
                    self.varIndexFlag[var][year] = -999

                if value != "skip":
                    info("variable %16s (%7s): index for year=%s => %4i (len = %2i) (flag index = %4i)" % 
                         (var,self.varType[var],year,self.varIndex[var][year],self.varLength[var],self.varIndexFlag[var][year]))
                else:
                    info("variable %16s (%7s): index for year=%s => skip" % (var,self.varType[var],year) )
                         
            firstYear = False

