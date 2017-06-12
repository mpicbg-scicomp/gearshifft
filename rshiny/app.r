## requires packages DT, shiny, ...

library(ggplot2)
library(shiny)

source("helper.r")

tmp1 <- ""
tmp2 <- ""

gearshifft_flist <- list.files(pattern = ".csv$", recursive = TRUE)


filter_by_tags <- function(flist, tags) {

    if( is.null(tags)==FALSE )
    {
        flist <- gearshifft_flist
        matches <- Reduce(intersect, lapply(tags, grep, flist))
        flist <- flist[ matches ]
    }
    return(flist)
}

get_input_files <- function(input,datapath=T) {

    if(input$sData1=='User')
        files <- ifelse(datapath, input$file1$datapath, input$file1$name)
    else {
        files <- input$file1
        tmp1 <<- input$file1
    }
    if(input$sData2=='User')
        files <- append(files, ifelse(datapath, input$file2$datapath, input$file2$name))
    else if(input$sData2=='gearshifft') {
        files <- append(files, input$file2)
        tmp2 <<- input$file2
    }


    return(unique(unlist(files)))
}

get_args <- function(input) {

    args <- get_args_default()
    args$placeness <- input$sInplace
    args$precision <- input$sPrec
    args$type <- input$sComplex
    args$kind <- input$sKind
    args$dim <- input$sDim
    args$xmetric <- input$sXmetric
    args$ymetric <- input$sYmetric
    args$notitle <- input$sNotitle
    args$run <- input$sRun
    return(args)

}

## Server

server <- function(input, output) {

    output$fInput1 <- renderUI({
        if (is.null(input$sData1))
            return()
        flist <- gearshifft_flist
        flist <- filter_by_tags(flist, input$tags1)
        if(grepl(tmp1, flist)==F)
            tmp1<<-""
        switch(input$sData1,
               "gearshifft" = selectInput("file1", "File", choices=flist, selected=tmp1),
               "User" = fileInput("file1", "File")
               )
    })

    output$fInput2 <- renderUI({
        if (is.null(input$sData2) || input$sData2=="none")
            return()
        flist <- gearshifft_flist
        flist <- filter_by_tags(flist, input$tags2)
        if(grepl(tmp2, flist)==F)
            tmp2<<-""
        switch(input$sData2,
               "gearshifft" = selectInput("file2", "File", choices=flist, selected=tmp2),
               "User" = fileInput("file2", "File")
               )
    })

    output$sTable <- DT::renderDataTable(DT::datatable({

        if(is.null(input$file1))
            return()
        input_files <- get_input_files(input)
        args <- get_args(input)

        df_data <- get_gearshifft_data(input_files)
        result <- get_gearshifft_tables(df_data, args)

        return(result$reduced)
    }, style="bootstrap"))

    output$sTableRaw <- DT::renderDataTable(DT::datatable({

        if(is.null(input$file1))
            return()
        input_files <- get_input_files(input)

        df_data <- get_gearshifft_data(input_files)

        return(df_data)
    }, style="bootstrap"))

    output$sPlot <- renderPlot({

        if(is.null(input$file1))
            return()
        input_files <- get_input_files(input)
        args <- get_args(input)

        df_data <- get_gearshifft_data(input_files)
        tables <- get_gearshifft_tables(df_data, args)

        ## aesthetics
        aes <- c()
        if(nlevels(as.factor(tables$reduced$hardware))>1)
            aes <- append(aes,"hardware")
        if(input$sKind=="-")
            aes <- append(aes,"kind")
        if(input$sAes!="-")
            aes <- append(aes,input$sAes)
        if(length(aes)<3)
            aes <- append(aes,"library")
        aes_str <- paste(aes, collapse=",")

        freqpoly <- F
        usepointsraw <- F
        usepoints <- F
        noerrorbar <- F

        ## plot type
        if(input$sPlotType=="Histogram") {
            freqpoly <- T
            noerrorbar <- T
        } else if(input$sPlotType=="Points") {
            usepointsraw <- T
        } else {
            usepoints <- input$sUsepoints || length(aes)>2
            noerrorbar <- input$sNoerrorbar
        }

        plot_gearshifft(tables,
                        aesthetics = aes_str,
                        logx = input$sLogx,
                        logy = input$sLogy,
                        freqpoly = freqpoly,
                        bins = input$sHistBins,
                        usepoints = usepoints,
                        usepointsraw = usepointsraw,
                        noerrorbar = noerrorbar
                        )
    })

    output$sPlotOptions <- renderUI({
        if(input$sPlotType == "Histogram")
            column(2, numericInput("sHistBins", "Bins", 200, min=10, max=1000))
        else if(input$sPlotType == "Lines") {
            fluidRow(column(1, checkboxInput("sUsepoints", "Draw Points")),
                     column(2, checkboxInput("sNoerrorbar", "Disable Error-Bars")))
        }
    })

    output$sInfo <- renderUI({
        input_files <- get_input_files(input)
        header <- get_gearshifft_header( input_files[1] )
        output$table1 <- renderTable({
            header$table1
        })
        output$table2 <- renderTable({
            header$table2
        })

        if(length(input_files)>1) {
            header2 <- get_gearshifft_header( input_files[2] )
            output$table3 <- renderTable({
                header2$table1
            })
            output$table4 <- renderTable({
                header2$table2
            })
            wellPanel(
                br(),
                h4(input_files[1]),
                fluidRow(
                    column(4, tableOutput("table1")),
                    column(4, tableOutput("table2"))
                ),
                h4(input_files[2]),
                fluidRow(
                    column(4, tableOutput("table3")),
                    column(4, tableOutput("table4"))
                )
            )
        } else {

            wellPanel(
                br(),
                h4(input_files[1]),
                fluidRow(
                    column(4, tableOutput("table1")),
                    column(4, tableOutput("table2"))
                )
            )
        }
    })

    #
    output$sHint <- renderUI({
        if(input$sPlotType == "Histogram")
            p("Histograms help to analyze data of the validation code.", HTML("<ul><li>Use Time_* as xmetric for the x axis.</li><li>Probably better to disable log-scaling</li><li>If you do not see any curves then disable some filters.</li></ul>"))
        else if(input$sPlotType == "Lines")
            p("Lines are drawn by the averages including error bars.", HTML("<ul><li>If you see jumps then you should enable more filters or use the 'Inspect' option.</li><li>Points are always drawn when the degree of freedom in the diagram is greater than 2.</li></ul>"))
        else if(input$sPlotType == "Points")
            p("This plot type allows to analyze the raw data by plotting each measure point. It helps analyzing the results of the validation code.")

    })
}






## User Interface

time_columns <- c("Time_Total","Time_FFT","Time_iFFT", "Time_Download", "Time_Upload", "Time_Allocation", "Time_PlanInitFwd", "Time_PlanInitInv", "Time_PlanDestroy")

ui <- fluidPage(

    theme="simplex.min.css",
    tags$style(type="text/css",
               "label {font-size: 12px;}",
               "p {font-weight: bold;}",
               "h3 {margin-top: 0px;}",
               ".checkbox {vertical-align: top; margin-top: 0px; padding-top: 0px;}"
               ),

    h1("gearshifft | Benchmark Analysis Tool"),
    p("gearshifft is an FFT benchmark suite to evaluate the performance of various FFT libraries on different architectures. Get ",
      a(href="https://github.com/mpicbg-scicomp/gearshifft/", "gearshifft on github.")),
    hr(),

    wellPanel(
        h3("Data"),
        p("Data is provided either by gearshifft or by uploaded csv files generated with gearshifft."),
        fluidRow(
            column(6, wellPanel( fluidRow(
                          column(3, selectInput("sData1", "Data 1", c("gearshifft", "User"))),
                          column(9, uiOutput("fInput1"))
                      ),
                      fluidRow(
                          checkboxGroupInput("tags1", "Tags",
                                             c("cuda"="cuda",
                                               "clfft"="clfft",
                                               "fftw"="fftw",
                                               "k80"="K80",
                                               "gtx1080"="GTX1080",
                                               "p100"="P100",
                                               "haswell"="haswell",
                                               "broadwell"="broadwell"),
                                             inline=T
                                             )))),
            column(6, wellPanel( fluidRow(
                          column(3, selectInput("sData2", "Data 2", c("gearshifft", "User", "none"), selected="none")),
                          column(9, uiOutput("fInput2"))
                      ),
                      fluidRow(
                          checkboxGroupInput("tags2", "Tags",
                                             c("cuda"="cuda",
                                               "clfft"="clfft",
                                               "fftw"="fftw",
                                               "k80"="K80",
                                               "gtx1080"="GTX1080",
                                               "p100"="P100",
                                               "haswell"="haswell",
                                               "broadwell"="broadwell"),
                                             inline=T
                                             ))))
        ),

        h3("Filtered by"),
        fluidRow(
            column(2, selectInput("sInplace", "Placeness", c("-","Inplace","Outplace"),selected="Inplace")),
            column(2, selectInput("sComplex", "Complex", c("-","Complex","Real"), selected="Real")),
            column(1, selectInput("sPrec", "Precision", c("-","float","double"), selected="-")),
            column(2, selectInput("sKind", "Kind", c("-","powerof2","radix357","oddshape"), selected="powerof2")),
            column(1, selectInput("sDim", "Dim", c("-","1","2","3"), selected="1")),
            column(2, selectInput("sXmetric", "xmetric", append(c("nbytes","id"),time_columns))),
            column(2, selectInput("sYmetric", "ymetric", time_columns))
        ),
        fluidRow(
            column(2, selectInput("sAes", "Inspect", c("-","inplace","flags","precision","dim"), selected="precision")),
            column(2, selectInput("sRun", "Run", c("-","Success", "Warmup"), selected="Success"))
        )
    ),

    tabsetPanel(
        ## Plot panel
        tabPanel("Plot",

                 br(),
                 plotOutput("sPlot"),
                 br(),
                 wellPanel(
                     h3("Plot Options"),
                     fluidRow(
                         column(3, selectInput("sPlotType", "Plot type", c("Lines","Histogram","Points"), selected="Lines")),
                         column(1, selectInput("sLogx", "Log-X", c("-","2","10"), selected="2")),
                         column(1, selectInput("sLogy", "Log-Y", c("-","2","10"), selected="10")),
                         column(1, checkboxInput("sNotitle", "Disable Title")),
                         uiOutput("sPlotOptions")
                     ),
                     uiOutput("sHint"))),
        ## Table panel
        tabPanel("Table",
                 
                 br(),
                 DT::dataTableOutput("sTable"),
                 p("A table aggregates the data and shows the average of the runs for each benchmark."),
                 div(HTML("<ul><li>xmoi: xmetric of interest (xmetric='nbytes' -> signal size in MiB)</li><li>ymoi: ymetric of interest</li></ul>"))
                 ),
        ## Table panel
        tabPanel("Raw Data",
                 
                 br(),
                 DT::dataTableOutput("sTableRaw")
                 ),
        tabPanel("Info",
                 
                 br(),
                 uiOutput("sInfo")                 
                 )
    ),
    hr(),
    
    ## fluidRow(verbatimTextOutput("log"))
    ##    mainPanel(plotOutput("distPlot"))
    ##  )
    
    span("This tool is powered by R Shiny Server.")
)

## will look for ui.R and server.R when reloading browser page, so you have to run
## R -e "shiny::runApp('~/shinyapp')"
shinyApp(ui = ui, server = server)
