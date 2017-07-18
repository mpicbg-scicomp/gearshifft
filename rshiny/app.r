## requires packages DT, shiny, ...

library(ggplot2)
library(shiny)

source("helper.r")

flist1_selected <- ""
flist2_selected <- ""
tmpid <- 0

gearshifft_flist <- list.files(pattern = ".csv$", recursive = TRUE)

create_filter <- function(filter_presets, inplace, complex, precision, kind, dim, xmetric, ymetric, inspect, run, ratio, plot="Lines", logx="2", logy="10") {

    tmpid <<- tmpid + 1

    preset <- list()
    preset$.id <- paste0(tmpid,":")
    preset$inplace <- inplace
    preset$complex <- complex
    preset$precision <- precision
    preset$kind <- kind
    preset$dim <- dim
    preset$run <- run
    preset$.sep0 <- "|"
    preset$inspect <- inspect
    preset$xmetric <- xmetric
    preset$ymetric <- ymetric
    preset$ratio <- ratio
    preset$.sep1 <- "|"
    preset$plot <- plot
    preset$logx <- logx
    preset$logy <- logy

    return(bind_rows(filter_presets, preset))
}

filter_to_string <- function(filter) {
    filter$dim <- paste0(filter$dim,"D")
    filter$run[ filter$run == "-" ] <- "| (incl. warmups)"
    filter$inspect <- ifelse( filter$inspect=="-", "-", paste0("$",filter$inspect) )
    filter$ratio <- ifelse( filter$ratio=="1", "(rel. to total time)", "" )
    filter$logx <- ifelse( filter$logx=="-", "-", paste0("logx=",filter$logx) )
    filter$logy <- ifelse( filter$logy=="-", "-", paste0("logy=",filter$logy) )
    str <- do.call(paste, filter)
    str <- gsub(" -", "", str)
    str <- gsub(" Success", "", str)
    return(str)
}

string_to_filter <- function(str) {
    id <- strtoi(unlist(strsplit(str, split=": "))[1])
    if(is.na(id))
        return(NULL)
    return(filter_presets[id,])
}

filter_presets_lines <- list()
filter_presets_lines <- create_filter(filter_presets_lines, "Inplace", "Real", precision="-", kind="powerof2", dim="1", xmetric="nbytes", ymetric="Time_Total", inspect="precision", run="Success", ratio="0")
filter_presets_lines <- create_filter(filter_presets_lines, "Inplace", "Real", "-", "powerof2", "1", "nbytes", "Time_Total", "precision", run="-", ratio="0")
filter_presets_lines <- create_filter(filter_presets_lines, "Inplace", "Real", "-", "powerof2", "1", "nbytes", "Time_FFT", "precision", "Success", ratio="0")
filter_presets_lines <- create_filter(filter_presets_lines, "Inplace", "Real", "-", "powerof2", "1", "nbytes", "Time_PlanInitFwd", "precision", "Success", ratio="1", "Lines", logx="2", logy="-")
filter_presets_lines <- create_filter(filter_presets_lines, "Inplace", "Real", "-", "powerof2", "1", "nbytes", "Size_DeviceBuffer+Size_DevicePlan", "precision", "Success", ratio="0")

filter_presets_hist <- list()
filter_presets_hist <- create_filter(filter_presets_hist, "Inplace", "Real", "-", "powerof2", "1", "Time_Total", "Time_Total", "precision", "Success", "0", "Histogram")
filter_presets_hist <- create_filter(filter_presets_hist, "Inplace", "Real", "-", "powerof2", "3", "Time_FFT", "Time_FFT", "precision", "Success", "0", "Histogram")

filter_presets_points <- list()
filter_presets_points <- create_filter(filter_presets_points, "Inplace", "Real", "-", "powerof2", "1", "id", "Time_Total", "precision", "Success", "0", "Points")
filter_presets_points <- create_filter(filter_presets_points, "Inplace", "Real", "-", "powerof2", "3", "id", "Time_FFT", "precision", "Success", "0", "Points")

filter_presets <- bind_rows(filter_presets_lines, filter_presets_hist, filter_presets_points)

## strings for the selectInput widget
filter_presets_gui <- list()
## to string
filter_presets_gui[['Lines']] <- filter_to_string(filter_presets_lines)
filter_presets_gui[['Histogram']] <- filter_to_string(filter_presets_hist)
filter_presets_gui[['Points']] <- filter_to_string(filter_presets_points)

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
        flist1_selected <<- input$file1
    }
    if(input$sData2=='User')
        files <- append(files, ifelse(datapath, input$file2$datapath, input$file2$name))
    else if(input$sData2=='gearshifft') {
        files <- append(files, input$file2)
        flist2_selected <<- input$file2
    }


    return(unique(unlist(files)))
}

get_args <- function(input) {

    args <- get_args_default()
    args$inplace <- input$sInplace
    args$complex <- input$sComplex
    args$precision <- input$sPrec
    args$kind <- input$sKind
    args$dim <- input$sDim
    args$xmetric <- input$sXmetric
    args$ymetric <- input$sYmetric
    args$notitle <- input$sNotitle
    args$run <- input$sRun
    if(input$sYRatio) {
        args$ymetric <- paste0(args$ymetric,"/Time_Total")
    }
    args$speedup <- input$sSpeedup
    return(args)

}

## Server

server <- function(input, output, session) {
    observe({
        filter <- string_to_filter(input$sFilter)
        if(is.null(filter)==FALSE) {
            updateSelectInput(session, "sInplace", selected = filter$inplace)
            updateSelectInput(session, "sComplex", selected = filter$complex)
            updateSelectInput(session, "sPrec", selected = filter$precision)
            updateSelectInput(session, "sKind", selected = filter$kind)
            updateSelectInput(session, "sDim", selected = filter$dim)
            updateSelectInput(session, "sXmetric", selected = filter$xmetric)
            updateSelectInput(session, "sYmetric", selected = filter$ymetric)
            updateSelectInput(session, "sAes", selected = filter$inspect)
            updateSelectInput(session, "sRun", selected = filter$run)
            updateCheckboxInput(session, "sYRatio", value = strtoi(filter$ratio))
            updateSelectInput(session, "sPlotType", selected = filter$plot)
            updateSelectInput(session, "sLogx", selected = filter$logx)
            updateSelectInput(session, "sLogy", selected = filter$logy)
        }
    })
    
    observe({
        if (input$sSpeedup==TRUE && (is.null(input$sData2) || input$sData2=="none"))
            updateCheckboxInput(session, "sSpeedup", value=FALSE)
    })
    
    output$fInput1 <- renderUI({
        if (is.null(input$sData1))
            return()
        flist <- gearshifft_flist
        flist <- filter_by_tags(flist, input$tags1) ## files matching tags like cuda p100 ...
        if(flist1_selected %in% flist == FALSE) ## if flist1_selected is not in (filtered) flist, disable it
            flist1_selected<<-""
        switch(input$sData1,
               "gearshifft" = selectInput("file1", "File", choices=flist, selected=flist1_selected),
               "User" = fileInput("file1", "File")
               )
    })

    output$fInput2 <- renderUI({
        if (is.null(input$sData2) || input$sData2=="none")
            return()
        flist <- gearshifft_flist
        flist <- filter_by_tags(flist, input$tags2)
        if(flist2_selected %in% flist == FALSE)
            flist2_selected<<-""
        switch(input$sData2,
               "gearshifft" = selectInput("file2", "File", choices=flist, selected=flist2_selected),
               "User" = fileInput("file2", "File")
               )
    })

    output$sTable <- DT::renderDataTable(DT::datatable({

        if(is.null(input$file1))
            return()
        input_files <- get_input_files(input)
        args <- get_args(input)

        df_data <- get_gearshifft_data(input_files,c(input$sCustomName1,input$sCustomName2))
        result <- get_gearshifft_tables(df_data, args)

        return(result$reduced)
    }, style="bootstrap"))

    output$sTableRaw <- DT::renderDataTable(DT::datatable({

        if(is.null(input$file1))
            return()
        input_files <- get_input_files(input)

        df_data <- get_gearshifft_data(input_files,c(input$sCustomName1,input$sCustomName2))

        return(df_data)
    }, style="bootstrap"))

    output$sPlot <- renderPlot({

        if(is.null(input$file1)) {
            return()
        }
        input_files <- get_input_files(input)
        args <- get_args(input)

        df_data <- get_gearshifft_data(input_files,c(input$sCustomName1,input$sCustomName2))
        tables <- get_gearshifft_tables(df_data, args)

        ## aesthetics
        aes <- c()
        if(nlevels(as.factor(tables$reduced$hardware))>1)
            aes <- append(aes,"hardware")
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
        if(input$sSpeedup==T) {
            noerrorbar <- T
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
                                               "haswell"="haswell"),
                                             inline=T
                                             )),
                      fluidRow(
                          column(8,textInput("sCustomName1","Custom Library Name (leave it empty for default label)",""))
                      )
                      )),
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
                                               "haswell"="haswell"),
                                             inline=T
                                             )),
                      fluidRow(
                          column(8,textInput("sCustomName2","Custom Library Name (leave it empty for default label)",""))
                      )
                      ))
        ),

        h3("Filtered by"),
        tabsetPanel(id="sFilterMask",
            tabPanel("Presets", br(),
                     fluidRow(column(6,
                       selectInput("sFilter", "Preset",
                                   filter_presets_gui
                                   )))),
            tabPanel("Custom", br(),
        fluidRow(
            column(2, selectInput("sInplace", "Placeness", c("-","Inplace","Outplace"),selected="Inplace")),
            column(2, selectInput("sComplex", "Complex", c("-","Complex","Real"), selected="Real")),
            column(1, selectInput("sPrec", "Precision", c("-","float","double"), selected="-")),
            column(2, selectInput("sKind", "Kind", c("-","powerof2","radix357","oddshape"), selected="powerof2")),
            column(1, selectInput("sDim", "Dim", c("-","1","2","3"), selected="1")),
            column(2, selectInput("sXmetric", "xmetric", append(c("nbytes","id"),time_columns))),
            column(2, selectInput("sYmetric", "ymetric", append(time_columns,c("Size_DeviceBuffer","Size_DevicePlan","Size_DeviceBuffer+Size_DevicePlan","Size_DeviceTransfer")), selected="Time_Total"))
        ),
        fluidRow(
            column(2, selectInput("sAes", "Inspect", c("-","inplace","flags","precision","dim","kind"), selected="precision")),
            column(2, selectInput("sRun", "Run", c("-","Success", "Warmup"), selected="Success")),
            column(2, checkboxInput("sYRatio","Ratio Total Time")),
            column(2, checkboxInput("sSpeedup","Speedup"))
        )))
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
