def printProgressBar(title, iteration, total, decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r|%s| Progress: |%s| %s%% Complete' %
          (title, bar, percent), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
