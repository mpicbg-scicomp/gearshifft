aggr_runs <- function(df, n, fun){
    aggregate(x = df,
              by = list(gl(ceiling(nrow(df)/n), n)[1:nrow(df)]),
              FUN = fun)
}

evalfiles <- function(csvfiles) {
    result <- list()
    for(csvfile in csvfiles) {
        ## import
        r<-read.table(csvfile,
                      sep=",",
                      quote='"',
                      header=T,
                      skip=3)
        ## only successful runs
        r<-r[r[11]=="Success",]

        ## collapse benchmark runs : static column body
        tmp <- r[seq(1,nrow(r),5),1:9]
        total <- tmp$nx * ifelse(tmp$ny>0, tmp$ny, 1) * ifelse(tmp$nz>0, tmp$nz, 1)
        body_aggr <- cbind(tmp, total)

        ## collapse benchmark runs : mean
        tmp <- aggr_runs(r[,12:ncol(r)], nruns, mean)
        ## union of benchmark data and aggregated runs
        ## excluding run and success column
        df.means<-cbind(body_aggr,tmp)

        ## bandwidth in MiB/s
        tmp <- df.means[,grepl( "Transfer", names(df.means) )] / 1048.576 /
            df.means[,grepl( "Upload|Download", names(df.means) )]
        bw.0 <- cbind(body_aggr,tmp)
        ## further aggregated bandwidth
        bw.1 <- aggregate(x=tmp,
                          by=list(bw.0$inplace,bw.0$complex,bw.0$precision,bw.0$dim),
                          FUN=mean)
        ## even further aggregation, collapse dim column
        bw.2 <- aggregate(x=tmp,
                          by=list(bw.0$inplace,bw.0$complex,bw.0$precision),
                          FUN=mean)
        ## times
        powerof2 <- df.means[df.means$kind=="powerof2",]
        oddshape <- df.means[df.means$kind=="oddshape",]
        radix357 <- df.means[df.means$kind=="radix357",]
        
        lib <- tolower(as.character(r[1,1]))
        print(lib)
        result[[lib]] <- list(bw.0=bw.0,
                    bw.1=bw.1,
                    bw.2=bw.2,
                    powerof2=powerof2,
                    oddshape=oddshape,
                    radix357=radix357)
    }
    return(result)
}

colours <- c("red", "orange", "blue", "yellow", "green")
nruns <- 5 # number of benchmark runs
csvfiles <- c("../results/K80/cuda-7.5.18/clfft_gcc5.3.0_RHEL6.7.csv",
              "../results/K80/cuda-7.5.18/cufft_gcc5.3.0_RHEL6.7.csv")
k80 <- evalfiles(csvfiles)
## join columns inplace, complex, precision into one column
fftypes <- do.call(paste,c(k80[[1]]$bw.2[,1:3],sep=" "))
## prepare data matrix for barchart
bw<-as.matrix(cbind(k80[[1]]$bw.2[4],
                    k80[[2]]$bw.2[4]))
colnames(bw)<-names(k80)
rownames(bw)<-fftypes

barplot(t(bw),
        main="Average Bandwidth on K80",
        xlab="FFT type",
        ylab="MiB/s",
        col=colours[1:2],
        legend = colnames(bw),
        beside=TRUE)
