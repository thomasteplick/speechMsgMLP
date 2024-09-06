/*
Neural Network (nn) using multilayer perceptron architecture
and the backpropagation algorithm.  This is a web application that uses
the html/template package to create the HTML.
The URL is http://127.0.0.1:8080/speechMLP.  There are two phases of
operation:  the training phase and the testing phase.  Epochs consising of
a sequence of examples are used to train the nn.  Each example consists
of an input vector of (x,y) coordinates and a desired class output.  The nn
itself consists of an input layer of nodes, one or more hidden layers of nodes,
and an output layer of nodes.  The nodes are connected by weighted links.  The
weights are trained by back propagating the output layer errors forward to the
input layer.  The chain rule of differential calculus is used to assign credit
for the errors in the output to the weights in the hidden layers.
The output layer outputs are subtracted from the desired to obtain the error.
The user trains first and then tests.

This application classifies audio wav files.  The Power Spectral Density of
each file is calculated and the magnitude squared is the input to the MLP.
The mean is removed and the magnitude squared is normalized to one.
The MLP classifies the wav file based on its spectral content.  The test
results are shown.  The user can plot the time domain or frequency domain
of the wav file.
*/

package main

import (
	"bufio"
	"fmt"
	"html/template"
	"log"
	"math"
	"math/cmplx"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/fft"
)

const (
	addr                 = "127.0.0.1:8080"               // http server listen address
	fileTrainingMLP      = "templates/trainingMLP.html"   // html for training MLP
	fileTestingMLP       = "templates/testingMLP.html"    // html for testing MLP
	fileVocabularyMLP    = "templates/vocabularyMLP.html" // html for adding words to the vocabulary
	patternTrainingMLP   = "/speechMsgMLP"                // http handler for training the MLP
	patternTestingMLP    = "/speechMsgMLPtest"            // http handler for testing the MLP
	patternVocabularyMLP = "/speechMsgMLPvocabulary"      // http handler for adding words to the vocabulary
	xlabels              = 11                             // # labels on x axis
	ylabels              = 11                             // # labels on y axis
	fileweights          = "weights.csv"                  // mlp weights
	a                    = 1.7159                         // activation function const
	b                    = 2.0 / 3.0                      // activation function const
	K1                   = b / a
	K2                   = a * a
	dataDir              = "data/"       // directory for the weights and audio wav files
	maxSamples           = 50000         // max audio wav samples > 5 sec * sampleRate
	classes              = 8             // number of audio wav files to classify
	rows                 = 300           // rows in canvas
	cols                 = 300           // columns in canvas
	sampleRate           = 8000          // Hz
	twoPi                = 2.0 * math.Pi // 2Pi
	bitDepth             = 16            // audio wav encoder/decoder sample size
	msgTestWav           = "message.wav" // Test message wav file
)

// Type to contain all the HTML template actions
type PlotT struct {
	Grid         []string // plotting grid
	Status       string   // status of the plot
	Xlabel       []string // x-axis labels
	Ylabel       []string // y-axis labels
	HiddenLayers string   // number of hidden layers
	LayerDepth   string   // number of Nodes in hidden layers
	Classes      string   // constant number of classes = 32
	LearningRate string   // size of weight update for each iteration
	Momentum     string   // previous weight update scaling factor
	Epochs       string   // number of epochs
	TestResults  string   // test message
	FFTSize      string   // 8192, 4098, 2048, 1024
	FFTWindow    string   // Bartlett, Welch, Hamming, Hanning, Rectangle
	TimeDomain   bool     // plot time domain, otherwise plot frequency domain
	Vocabulary   []string // vocabulary for the message
}

// Type to hold the minimum and maximum data values of the MSE in the Learning Curve
type Endpoints struct {
	xmin float64
	xmax float64
	ymin float64
	ymax float64
}

// graph node
type Node struct {
	y     float64 // output of this node for forward prop
	delta float64 // local gradient for backward prop
}

// graph links
type Link struct {
	wgt      float64 // weight
	wgtDelta float64 // previous weight update used in momentum
}

// training examples
type Sample struct {
	name    string    // audio wav frequency content
	desired int       // numerical class of the audio wav file
	psd     []float64 // Power Spectral Density
}

type Bound struct {
	start, stop int // word boundaries in the message
}

// Primary data structure for holding the MLP Backprop state
type MLP struct {
	plot         *PlotT   // data to be distributed in the HTML template
	Endpoints             // embedded struct
	link         [][]Link // links in the graph
	node         [][]Node // nodes in the graph
	samples      []Sample
	nsamples     int       // number of audio wav samples
	mse          []float64 // mean square error in output layer per epoch used in Learning Curve
	epochs       int       // number of epochs
	learningRate float64   // learning rate parameter
	momentum     float64   // delta weight scale constant
	hiddenLayers int       // number of hidden layers
	desired      []float64 // desired output of the sample
	layerDepth   int       // hidden layer number of nodes
	fftSize      int       // 1024, 2048, 4096, 8192
	fftWindow    string    // one of the winTypes
	words        []string  // words in test message
	wordWindow   int       // message word window to accumulate audio level
	dbLevel      int       // message word audio level to determine start
}

// Window function type
type Window func(n int, m int) complex128

// global variables for parse and execution of the html template
var (
	tmplTrainingMLP   *template.Template
	tmplTestingMLP    *template.Template
	tmplVocabularyMLP *template.Template
	winType           = []string{"Bartlett", "Welch", "Hamming", "Hanning", "Rectangle"}
)

// Bartlett window
func bartlett(n int, m int) complex128 {
	real := 1.0 - math.Abs((float64(n)-float64(m))/float64(m))
	return complex(real, 0)
}

// Welch window
func welch(n int, m int) complex128 {
	x := math.Abs((float64(n) - float64(m)) / float64(m))
	real := 1.0 - x*x
	return complex(real, 0)
}

// Hamming window
func hamming(n int, m int) complex128 {
	return complex(.54-.46*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Hanning window
func hanning(n int, m int) complex128 {
	return complex(.5-.5*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Rectangle window
func rectangle(n int, m int) complex128 {
	return 1.0
}

// calculateMSE calculates the MSE at the output layer every epoch
func (mlp *MLP) calculateMSE(epoch int) {
	// loop over the output layer nodes
	var err float64 = 0.0
	outputLayer := mlp.hiddenLayers + 1
	for n := 0; n < len(mlp.node[outputLayer]); n++ {
		// Calculate (desired[n] - mlp.node[L][n].y)^2 and store in mlp.mse[n]
		err = float64(mlp.desired[n]) - mlp.node[outputLayer][n].y
		err2 := err * err
		mlp.mse[epoch] += err2
	}
	mlp.mse[epoch] /= float64(classes)

	// calculate min/max mse
	if mlp.mse[epoch] < mlp.ymin {
		mlp.ymin = mlp.mse[epoch]
	}
	if mlp.mse[epoch] > mlp.ymax {
		mlp.ymax = mlp.mse[epoch]
	}
}

// determineClass determines testing example class given sample number and sample
func (mlp *MLP) determineClass() error {
	// At output layer, classify example and store the name

	// convert node outputs to the class; zero is the threshold
	class := 0
	for i, output := range mlp.node[mlp.hiddenLayers+1] {
		if output.y > 0.0 {
			class |= (1 << i)
		}
	}
	mlp.words = append(mlp.words, mlp.samples[class].name)

	return nil
}

// class2desired constructs the desired output from the given class
func (mlp *MLP) class2desired(class int) {
	// tranform int to slice of -1 and 1 representing the 0 and 1 bits
	for i := 0; i < len(mlp.desired); i++ {
		if class&1 == 1 {
			mlp.desired[i] = 1
		} else {
			mlp.desired[i] = -1
		}
		class >>= 1
	}
}

func (mlp *MLP) propagateForward(samp Sample) error {
	// Assign sample to input layer, i=0 is the bias equal to one
	for i := 1; i < len(mlp.node[0]); i++ {
		mlp.node[0][i].y = float64(samp.psd[i-1])
	}

	// calculate desired from the class
	mlp.class2desired(samp.desired)

	// Loop over layers: mlp.hiddenLayers + output layer
	// input->first hidden, then hidden->hidden,..., then hidden->output
	for layer := 1; layer <= mlp.hiddenLayers; layer++ {
		// Loop over nodes in the layer, d1 is the layer depth of current
		d1 := len(mlp.node[layer])
		for i1 := 1; i1 < d1; i1++ { // this layer loop
			// Each node in previous layer is connected to current node because
			// the network is fully connected.  d2 is the layer depth of previous
			d2 := len(mlp.node[layer-1])
			// Loop over weights to get v
			v := 0.0
			for i2 := 0; i2 < d2; i2++ { // previous layer loop
				v += mlp.link[layer-1][i2*(d1-1)+i1-1].wgt * mlp.node[layer-1][i2].y
			}
			// compute output y = Phi(v)
			mlp.node[layer][i1].y = a * math.Tanh(b*v)
		}
	}

	// last layer is different because there is no bias node, so the indexing is different
	layer := mlp.hiddenLayers + 1
	d1 := len(mlp.node[layer])
	for i1 := 0; i1 < d1; i1++ { // this layer loop
		// Each node in previous layer is connected to current node because
		// the network is fully connected.  d2 is the layer depth of previous
		d2 := len(mlp.node[layer-1])
		// Loop over weights to get v
		v := 0.0
		for i2 := 0; i2 < d2; i2++ { // previous layer loop
			v += mlp.link[layer-1][i2*d1+i1].wgt * mlp.node[layer-1][i2].y
		}
		// compute output y = Phi(v)
		mlp.node[layer][i1].y = a * math.Tanh(b*v)
	}

	return nil
}

func (mlp *MLP) propagateBackward() error {

	// output layer is different, no bias node, so the indexing is different
	// Loop over nodes in output layer
	layer := mlp.hiddenLayers + 1
	d1 := len(mlp.node[layer])
	for i1 := 0; i1 < d1; i1++ { // this layer loop
		//compute error e=d-Phi(v)
		mlp.node[layer][i1].delta = mlp.desired[i1] - mlp.node[mlp.hiddenLayers+1][i1].y
		// Multiply error by this node's Phi'(v) to get local gradient.
		mlp.node[layer][i1].delta *= K1 * (K2 - mlp.node[layer][i1].y*mlp.node[layer][i1].y)
		// Send this node's local gradient to previous layer nodes through corresponding link.
		// Each node in previous layer is connected to current node because the network
		// is fully connected.  d2 is the previous layer depth
		d2 := len(mlp.node[layer-1])
		for i2 := 0; i2 < d2; i2++ { // previous layer loop
			mlp.node[layer-1][i2].delta += mlp.link[layer-1][i2*d1+i1].wgt * mlp.node[layer][i1].delta
			// Compute weight delta, Update weight with momentum, y, and local gradient
			wgtDelta := mlp.learningRate * mlp.node[layer][i1].delta * mlp.node[layer-1][i2].y
			mlp.link[layer-1][i2*d1+i1].wgt +=
				wgtDelta + mlp.momentum*mlp.link[layer-1][i2*d1+i1].wgtDelta
			// update weight delta
			mlp.link[layer-1][i2*d1+i1].wgtDelta = wgtDelta

		}
		// Reset this local gradient to zero for next training example
		mlp.node[layer][i1].delta = 0.0
	}

	// Loop over layers in backward direction, starting at the last hidden layer
	for layer := mlp.hiddenLayers; layer > 0; layer-- {
		// Loop over nodes in this layer, d1 is the current layer depth
		d1 := len(mlp.node[layer])
		for i1 := 1; i1 < d1; i1++ { // this layer loop
			// Multiply deltas propagated from past node by this node's Phi'(v) to get local gradient.
			mlp.node[layer][i1].delta *= K1 * (K2 - mlp.node[layer][i1].y*mlp.node[layer][i1].y)
			// Send this node's local gradient to previous layer nodes through corresponding link.
			// Each node in previous layer is connected to current node because the network
			// is fully connected.  d2 is the previous layer depth
			d2 := len(mlp.node[layer-1])
			for i2 := 0; i2 < d2; i2++ { // previous layer loop
				mlp.node[layer-1][i2].delta += mlp.link[layer-1][i2*(d1-1)+i1-1].wgt * mlp.node[layer][i1].delta
				// Compute weight delta, Update weight with momentum, y, and local gradient
				// anneal learning rate parameter: mlp.learnRate/(epoch*layer)
				// anneal momentum: momentum/(epoch*layer)
				wgtDelta := mlp.learningRate * mlp.node[layer][i1].delta * mlp.node[layer-1][i2].y
				mlp.link[layer-1][i2*(d1-1)+i1-1].wgt +=
					wgtDelta + mlp.momentum*mlp.link[layer-1][i2*(d1-1)+i1-1].wgtDelta
				// update weight delta
				mlp.link[layer-1][i2*(d1-1)+i1-1].wgtDelta = wgtDelta

			}
			// Reset this local gradient to zero for next training example
			mlp.node[layer][i1].delta = 0.0
		}
	}
	return nil
}

// runEpochs performs forward and backward propagation over each sample
func (mlp *MLP) runEpochs() error {

	// number of directories containing the training wav files
	const ndirs int = 7
	// read order of the audio wav file directories
	readOrder := make([]int, ndirs)
	for i := 0; i < ndirs; i++ {
		readOrder[i] = i
	}

	// Initialize the weights

	// input layer
	// initialize the wgt and wgtDelta randomly, zero mean, normalize by fan-in
	for i := range mlp.link[0] {
		mlp.link[0][i].wgt = 2.0 * (rand.ExpFloat64() - .5) / float64(mlp.fftSize+1)
		mlp.link[0][i].wgtDelta = 2.0 * (rand.ExpFloat64() - .5) / float64(mlp.fftSize+1)
	}

	// output layer links
	for i := range mlp.link[mlp.hiddenLayers] {
		mlp.link[mlp.hiddenLayers][i].wgt = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
		mlp.link[mlp.hiddenLayers][i].wgtDelta = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
	}

	// hidden layers
	for lay := 1; lay < len(mlp.link)-1; lay++ {
		for link := 0; link < len(mlp.link[lay]); link++ {
			mlp.link[lay][link].wgt = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
			mlp.link[lay][link].wgtDelta = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
		}
	}
	for n := 0; n < mlp.epochs; n++ {
		// loop over the audio wav file directories
		// shuffle the audio wav file directories read order
		rand.Shuffle(len(readOrder), func(i, j int) {
			readOrder[i], readOrder[j] = readOrder[j], readOrder[i]
		})

		for _, val := range readOrder {
			wavdir := filepath.Join(dataDir, fmt.Sprintf("audiowav%d", val))
			if err := mlp.createExamples(wavdir); err != nil {
				fmt.Printf("createExamples error: %v\n", err)
				return fmt.Errorf("createExamples error: %v", err.Error())
			}

			// Shuffle training exmaples
			rand.Shuffle(len(mlp.samples), func(i, j int) {
				mlp.samples[i], mlp.samples[j] = mlp.samples[j], mlp.samples[i]
			})

			// Loop over the training examples
			for _, samp := range mlp.samples {
				// Forward Propagation
				err := mlp.propagateForward(samp)
				if err != nil {
					return fmt.Errorf("forward propagation error: %s", err.Error())
				}

				// Backward Propagation
				err = mlp.propagateBackward()
				if err != nil {
					return fmt.Errorf("backward propagation error: %s", err.Error())
				}
			}
		}
		// At the end of each epoch, loop over the output nodes and calculate mse
		mlp.calculateMSE(n)

	}
	return nil
}

// init parses the html template files
func init() {
	tmplTrainingMLP = template.Must(template.ParseFiles(fileTrainingMLP))
	tmplTestingMLP = template.Must(template.ParseFiles(fileTestingMLP))
	tmplVocabularyMLP = template.Must(template.ParseFiles(fileVocabularyMLP))
}

// createExamples creates a slice of training or testing examples
func (mlp *MLP) createExamples(wavdir string) error {
	// read in audio wav files and convert 16-bit samples to []float64
	files, err := os.ReadDir(wavdir)
	if err != nil {
		fmt.Printf("ReadDir for %s error: %v\n", wavdir, err)
		return fmt.Errorf("ReadDir for %s error %v", wavdir, err.Error())
	}

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	N := mlp.fftSize
	PSD := make([]float64, N/2)

	// Each audio wav file is a separate audio class
	class := 0
	for _, dirEntry := range files {
		name := dirEntry.Name()
		if filepath.Ext(name) == ".wav" {
			f, err := os.Open(path.Join(wavdir, name))
			if err != nil {
				fmt.Printf("Open %s error: %v\n", name, err)
				return fmt.Errorf("file Open %s error: %v", name, err.Error())
			}
			defer f.Close()
			// only process classes files
			if class == classes {
				return fmt.Errorf("can only process %v wav files", classes)
			}

			dec := wav.NewDecoder(f)
			bufInt := audio.IntBuffer{
				Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
				Data:   make([]int, maxSamples), SourceBitDepth: bitDepth}
			n, err := dec.PCMBuffer(&bufInt)
			if err != nil {
				fmt.Printf("PCMBuffer error: %v\n", err)
				return fmt.Errorf("PCMBuffer error: %v", err.Error())
			}
			bufFlt := bufInt.AsFloatBuffer()
			//fmt.Printf("%s samples = %d\n", name, n)
			mlp.nsamples = n

			// calculate the PSD using Bartlett's or Welch's variant of the Periodogram
			_, _, err = mlp.calculatePSD(bufFlt.Data, PSD, "normalize", mlp.fftWindow, mlp.fftSize)
			if err != nil {
				fmt.Printf("calculatePSD error: %v\n", err)
				return fmt.Errorf("calculatePSD error: %v", err.Error())
			}

			// save the name of the audio wav without the ext
			mlp.samples[class].name = strings.Split(name, ".")[0]
			// The desired output of the MLP is class
			mlp.samples[class].desired = class
			copy(mlp.samples[class].psd, PSD)
			class++
		}
	}

	return nil
}

// display the vocabulary so the user can compose a message
func fillVocabulary(plot *PlotT) error {
	// read audio wav files in dataDir\audiowav0 and convert the filename to class
	files, err := os.ReadDir(filepath.Join(dataDir, "audiowav0"))
	if err != nil {
		fmt.Printf("ReadDir for %s error: %v\n", filepath.Join(dataDir, "audiowav0"), err)
		return fmt.Errorf("ReadDir for %s error %v", filepath.Join(dataDir, "audiowav0"), err.Error())
	}

	plot.Vocabulary = make([]string, 0)
	// Each audio wav file is a separate audio class
	for _, dirEntry := range files {
		name := dirEntry.Name()
		if filepath.Ext(name) == ".wav" {
			plot.Vocabulary = append(plot.Vocabulary, strings.Split(name, ".")[0])
		}
	}
	return nil
}

// retrieveClasses retrieves the classes discovered during training
func (mlp *MLP) retrieveClasses() error {
	// read audio wav files in dataDir\audiowav and convert the filename to class
	files, err := os.ReadDir(filepath.Join(dataDir, "audiowav0"))
	if err != nil {
		fmt.Printf("ReadDir for %s error: %v\n", filepath.Join(dataDir, "audiowav0"), err)
		return fmt.Errorf("ReadDir for %s error %v", filepath.Join(dataDir, "audiowav0"), err.Error())
	}

	// Each audio wav file is a separate audio class
	class := 0
	for _, dirEntry := range files {
		name := dirEntry.Name()
		if filepath.Ext(name) == ".wav" {

			mlp.samples = append(mlp.samples, Sample{})
			// save the name of the audio wav without the ext
			mlp.samples[class].name = strings.Split(name, ".")[0]
			class++
		}
	}
	fmt.Printf("Read %d wav files for testing.\n", class)

	return nil
}

// newMLP constructs an MLP instance for training
func newMLP(r *http.Request, hiddenLayers int, plot *PlotT) (*MLP, error) {
	// Read the training parameters in the HTML Form

	txt := r.FormValue("layerdepth")
	layerDepth, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("layerdepth int conversion error: %v\n", err)
		return nil, fmt.Errorf("layerdepth int conversion error: %s", err.Error())
	}

	txt = r.FormValue("learningrate")
	learningRate, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("learningrate float conversion error: %v\n", err)
		return nil, fmt.Errorf("learningrate float conversion error: %s", err.Error())
	}

	txt = r.FormValue("momentum")
	momentum, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("momentum float conversion error: %v\n", err)
		return nil, fmt.Errorf("momentum float conversion error: %s", err.Error())
	}

	txt = r.FormValue("epochs")
	epochs, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("epochs int conversion error: %v\n", err)
		return nil, fmt.Errorf("epochs int conversion error: %s", err.Error())
	}

	fftWindow := r.FormValue("fftwindow")

	txt = r.FormValue("fftsize")
	fftSize, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("fftsize int conversion error: %v\n", err)
		return nil, fmt.Errorf("fftsize int conversion error: %s", err.Error())
	}

	mlp := MLP{
		hiddenLayers: hiddenLayers,
		layerDepth:   layerDepth,
		epochs:       epochs,
		learningRate: learningRate,
		momentum:     momentum,
		fftSize:      fftSize,
		fftWindow:    fftWindow,
		plot:         plot,
		Endpoints: Endpoints{
			ymin: math.MaxFloat64,
			ymax: -math.MaxFloat64,
			xmin: 0,
			xmax: float64(epochs - 1)},
		samples: make([]Sample, classes),
		words:   make([]string, 0),
	}
	for i := range mlp.samples {
		mlp.samples[i].psd = make([]float64, fftSize/2)
	}

	// construct link that holds the weights and weight deltas
	mlp.link = make([][]Link, hiddenLayers+1)

	// input layer
	mlp.link[0] = make([]Link, (fftSize/2+1)*layerDepth)

	// outer layer nodes
	olnodes := int(math.Ceil(math.Log2(float64(classes))))

	// output layer links
	mlp.link[len(mlp.link)-1] = make([]Link, olnodes*(layerDepth+1))

	// hidden layer links
	for i := 1; i < len(mlp.link)-1; i++ {
		mlp.link[i] = make([]Link, (layerDepth+1)*layerDepth)
	}

	// construct node, init node[i][0].y to 1.0 (bias)
	mlp.node = make([][]Node, hiddenLayers+2)

	// input layer
	mlp.node[0] = make([]Node, fftSize/2+1)
	// set first node in the layer (bias) to 1
	mlp.node[0][0].y = 1.0

	// output layer, which has no bias node
	mlp.node[hiddenLayers+1] = make([]Node, olnodes)

	// hidden layers
	for i := 1; i <= hiddenLayers; i++ {
		mlp.node[i] = make([]Node, layerDepth+1)
		// set first node in the layer (bias) to 1
		mlp.node[i][0].y = 1.0
	}

	// construct desired from classes, binary representation
	mlp.desired = make([]float64, olnodes)

	// mean-square error
	mlp.mse = make([]float64, epochs)

	return &mlp, nil
}

// gridFillInterp inserts the data points in the grid and draws a straight line between points
func (mlp *MLP) gridFillInterp() error {
	var (
		x            float64 = 0.0
		y            float64 = mlp.mse[0]
		prevX, prevY float64
		xscale       float64
		yscale       float64
	)

	// Mark the data x-y coordinate online at the corresponding
	// grid row/column.

	// Calculate scale factors for x and y
	xscale = float64(cols-1) / (mlp.xmax - mlp.xmin)
	yscale = float64(rows-1) / (mlp.ymax - mlp.ymin)

	mlp.plot.Grid = make([]string, rows*cols)

	// This cell location (row,col) is on the line
	row := int((mlp.ymax-y)*yscale + .5)
	col := int((x-mlp.xmin)*xscale + .5)
	mlp.plot.Grid[row*cols+col] = "online"

	prevX = x
	prevY = y

	// Scale factor to determine the number of interpolation points
	lenEPy := mlp.ymax - mlp.ymin
	lenEPx := mlp.xmax - mlp.xmin

	// Continue with the rest of the points in the file
	for i := 1; i < mlp.epochs; i++ {
		x++
		// ensemble average of the mse
		y = mlp.mse[i]

		// This cell location (row,col) is on the line
		row := int((mlp.ymax-y)*yscale + .5)
		col := int((x-mlp.xmin)*xscale + .5)
		mlp.plot.Grid[row*cols+col] = "online"

		// Interpolate the points between previous point and current point

		/* lenEdge := math.Sqrt((x-prevX)*(x-prevX) + (y-prevY)*(y-prevY)) */
		lenEdgeX := math.Abs((x - prevX))
		lenEdgeY := math.Abs(y - prevY)
		ncellsX := int(float64(cols) * lenEdgeX / lenEPx) // number of points to interpolate in x-dim
		ncellsY := int(float64(rows) * lenEdgeY / lenEPy) // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := ncellsX
		if ncellsY > ncells {
			ncells = ncellsY
		}

		stepX := (x - prevX) / float64(ncells)
		stepY := (y - prevY) / float64(ncells)

		// loop to draw the points
		interpX := prevX
		interpY := prevY
		for i := 0; i < ncells; i++ {
			row := int((mlp.ymax-interpY)*yscale + .5)
			col := int((interpX-mlp.xmin)*xscale + .5)
			mlp.plot.Grid[row*cols+col] = "online"
			interpX += stepX
			interpY += stepY
		}

		// Update the previous point with the current point
		prevX = x
		prevY = y
	}
	return nil
}

// insertLabels inserts x- an y-axis labels in the plot
func (mlp *MLP) insertLabels() {
	mlp.plot.Xlabel = make([]string, xlabels)
	mlp.plot.Ylabel = make([]string, ylabels)
	// Construct x-axis labels
	incr := (mlp.xmax - mlp.xmin) / (xlabels - 1)
	x := mlp.xmin
	// First label is empty for alignment purposes
	for i := range mlp.plot.Xlabel {
		mlp.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (mlp.ymax - mlp.ymin) / (ylabels - 1)
	y := mlp.ymin
	for i := range mlp.plot.Ylabel {
		mlp.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}
}

// handleTraining performs forward and backward propagation to calculate the weights
func handleTrainingMLP(w http.ResponseWriter, r *http.Request) {

	var (
		plot PlotT
		mlp  *MLP
	)

	// Get the number of hidden layers
	txt := r.FormValue("hiddenlayers")
	// Need hidden layers to continue
	if len(txt) > 0 {
		hiddenLayers, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Hidden Layers int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Hidden Layers conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// create MLP instance to hold state
		mlp, err = newMLP(r, hiddenLayers, &plot)
		if err != nil {
			fmt.Printf("newMLP() error: %v\n", err)
			plot.Status = fmt.Sprintf("newMLP() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Loop over the Epochs
		err = mlp.runEpochs()
		if err != nil {
			fmt.Printf("runEnsembles() error: %v\n", err)
			plot.Status = fmt.Sprintf("runEnsembles() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Put MSE vs Epoch in PlotT
		err = mlp.gridFillInterp()
		if err != nil {
			fmt.Printf("gridFillInterp() error: %v\n", err)
			plot.Status = fmt.Sprintf("gridFillInterp() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// insert x-labels and y-labels in PlotT
		mlp.insertLabels()

		// At the end of all epochs, insert form previous control items in PlotT
		mlp.plot.HiddenLayers = strconv.Itoa(mlp.hiddenLayers)
		mlp.plot.LayerDepth = strconv.Itoa(mlp.layerDepth)
		mlp.plot.Classes = strconv.Itoa(classes)
		mlp.plot.LearningRate = strconv.FormatFloat(mlp.learningRate, 'f', 4, 64)
		mlp.plot.Momentum = strconv.FormatFloat(mlp.momentum, 'f', 4, 64)
		mlp.plot.Epochs = strconv.Itoa(mlp.epochs)

		// Save hidden layers, hidden layer depth, classes, epochs, fft size, fft window,
		// and weights to csv file, one layer per line
		f, err := os.Create(path.Join(dataDir, fileweights))
		if err != nil {
			fmt.Printf("os.Create() file %s error: %v\n", path.Join(fileweights), err)
			plot.Status = fmt.Sprintf("os.Create() file %s error: %v", path.Join(fileweights), err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		defer f.Close()
		// save MLP parameters
		fmt.Fprintf(f, "%d,%d,%d,%d,%f,%f,%d,%s\n",
			mlp.epochs, mlp.hiddenLayers, mlp.layerDepth, classes, mlp.learningRate, mlp.momentum, mlp.fftSize, mlp.fftWindow)
		// save weights
		// save first layer, one weight per line because too long to scan in
		for _, node := range mlp.link[0] {
			fmt.Fprintf(f, "%.10f\n", node.wgt)
		}
		// save remaining layers one layer per line with csv
		for _, layer := range mlp.link[1:] {
			for _, node := range layer {
				fmt.Fprintf(f, "%.10f,", node.wgt)
			}
			fmt.Fprintln(f)
		}

		mlp.plot.Status = "MSE plotted"

		// Execute data on HTML template
		if err = tmplTrainingMLP.Execute(w, mlp.plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	} else {
		plot.Status = "Enter Multilayer Perceptron (MLP) training parameters."
		// Write to HTTP using template and grid
		if err := tmplTrainingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}

	}
}

// findWords finds the word boundaries in the testing message
func (mlp *MLP) findWords(data []float64) ([]Bound, error) {

	// prevent oscillation about threshold
	const hystersis = 0.9

	var (
		old    float64 = 0.0
		new    float64 = 0.0
		cur    int     = 0
		start  int     = 0
		stop   int     = 0
		sum    float64 = 0.0
		k      int     = 0
		j      int     = 0
		L      int     = mlp.nsamples
		max    float64 = 0.0
		bounds []Bound = make([]Bound, 0)
		avg    float64 = 0.0
	)

	// Find the maximum and normalize the data
	for i := 0; i < L; i++ {
		new = math.Abs(data[i])
		avg += new
		if new > max {
			max = new
		}
	}
	avg /= float64(L)

	// The number of samples in the audio level integration window.
	// Determines when the word and message ends
	// Convert wordWindow to ms
	win := int(float64(mlp.wordWindow) * .001 / (1.0 / float64(sampleRate)))
	// The word start audio level.
	level := max / math.Pow(10.0, float64(mlp.dbLevel)/20.0)
	// Minimum audio integration to determine when word begins and ends
	levelSum := float64(win) * avg
	buf := make([]float64, win)

	fmt.Printf("window = %d, level= %f, levelSum = %f, max = %f\n", win, level, levelSum, max)

	for k < L {
		for k < L {
			new = math.Abs(data[k])
			old = buf[cur]
			buf[cur] = new
			sum += (new - old)
			cur = (cur + 1) % win
			if k >= stop+win && sum > levelSum {
				start = k - win
				bounds = append(bounds, Bound{start: start})
				k++
				fmt.Printf("start = %.2f sec, sum = %.2f ", float64(start)*.000125, sum)
				break
			}
			k++
		}

		for k < L {
			new = math.Abs(data[k])
			old = buf[cur]
			buf[cur] = new
			sum += (new - old)
			cur = (cur + 1) % win
			if k > start+win && sum < levelSum*hystersis {
				stop = k
				bounds[j].stop = stop
				k++
				fmt.Printf("stop = %.2f sec, sum = %.2f\n", float64(stop)*.000125, sum)
				break
			}
			k++
		}
		j++
	}

	return bounds, nil
}

// Classify test examples and display test results
func (mlp *MLP) runClassification() error {

	// Open the testing message
	f, err := os.Open(filepath.Join(dataDir, msgTestWav))
	if err != nil {
		fmt.Printf("Open file %s error: %v", msgTestWav, err)
		return fmt.Errorf("open file %s error: %s", msgTestWav, err.Error())
	}
	defer f.Close()

	// Create wav Decoder, intBuf, fltBuf and Decode the wav file
	dec := wav.NewDecoder(f)
	bufInt := audio.IntBuffer{
		Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
		Data:   make([]int, maxSamples), SourceBitDepth: bitDepth}
	n, err := dec.PCMBuffer(&bufInt)
	if err != nil {
		fmt.Printf("PCMBuffer error: %v\n", err)
		return fmt.Errorf("PCMBuffer error: %v", err.Error())
	}
	bufFlt := bufInt.AsFloatBuffer()
	//fmt.Printf("%s samples = %d\n", filename, n)
	mlp.nsamples = n

	// loop over fltBuf and find the word boundaries
	bounds, err := mlp.findWords(bufFlt.Data)
	if err != nil {
		fmt.Printf("findWords error: %v", err)
		return fmt.Errorf("findWords error: %s", err.Error())
	}

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	N := mlp.fftSize
	PSD := make([]float64, N/2)

	// Loop over bounds and calculatePSD() for each word found in the test message
	// Propagate forward and classify the word's PSD
	// Insert the class in mlp.words using mlp.samples[class].name
	for _, bound := range bounds {
		// calculate the PSD using Bartlett's or Welch's variant of the Periodogram
		mlp.nsamples = bound.stop - bound.start
		_, _, err = mlp.calculatePSD(bufFlt.Data[bound.start:bound.stop], PSD, "normalize", mlp.fftWindow, mlp.fftSize)
		if err != nil {
			fmt.Printf("calculatePSD error: %v\n", err)
			return fmt.Errorf("calculatePSD error: %v", err.Error())
		}

		samp := Sample{desired: 0, psd: PSD, name: ""}
		err := mlp.propagateForward(samp)
		if err != nil {
			return fmt.Errorf("forward propagation error: %s", err.Error())
		}
		err = mlp.determineClass()
		if err != nil {
			return fmt.Errorf("determineClass error: %s", err.Error())
		}
	}
	mlp.nsamples = n

	mlp.plot.TestResults = strings.Join(mlp.words, " ")

	mlp.plot.LearningRate = strconv.FormatFloat(mlp.learningRate, 'f', -1, 64)
	mlp.plot.Momentum = strconv.FormatFloat(mlp.momentum, 'f', -1, 64)
	mlp.plot.HiddenLayers = strconv.Itoa(mlp.hiddenLayers)
	mlp.plot.LayerDepth = strconv.Itoa(mlp.layerDepth)
	mlp.plot.Classes = strconv.Itoa(classes)
	mlp.plot.FFTSize = strconv.Itoa(mlp.fftSize)
	mlp.plot.FFTWindow = mlp.fftWindow
	mlp.plot.Epochs = strconv.Itoa(mlp.epochs)

	mlp.plot.Status = "Testing results completed."

	return nil
}

// newTestingMLP constructs an MLP from the saved mlp weights and parameters
func newTestingMLP(r *http.Request, plot *PlotT) (*MLP, error) {
	// Read in weights from csv file, ordered by layers, and MLP parameters
	f, err := os.Open(path.Join(dataDir, fileweights))
	if err != nil {
		fmt.Printf("Open file %s error: %v", fileweights, err)
		return nil, fmt.Errorf("open file %s error: %s", fileweights, err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// get the parameters
	scanner.Scan()
	line := scanner.Text()

	items := strings.Split(line, ",")

	epochs, err := strconv.Atoi(items[0])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[0], err)
		return nil, err
	}

	hiddenLayers, err := strconv.Atoi(items[1])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[1], err)
		return nil, err
	}
	hidLayersDepth, err := strconv.Atoi(items[2])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[2], err)
		return nil, err
	}
	_, err = strconv.Atoi(items[3])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[3], err)
		return nil, err
	}

	learningRate, err := strconv.ParseFloat(items[4], 64)
	if err != nil {
		fmt.Printf("Conversion to float of %s error: %v\n", items[4], err)
		return nil, err
	}

	momentum, err := strconv.ParseFloat(items[5], 64)
	if err != nil {
		fmt.Printf("Conversion to float of %s error: %v\n", items[5], err)
		return nil, err
	}

	fftSize, err := strconv.Atoi(items[6])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[6], err)
		return nil, err
	}

	fftWindow := items[7]

	text := r.FormValue("window")
	if len(text) == 0 {
		return nil, fmt.Errorf("select Threshold and Window from the lists")
	}
	window, err := strconv.Atoi(text)
	if err != nil {
		fmt.Printf("Conversion to int of 'window' error: %v\n", err)
		return nil, err
	}

	text = r.FormValue("threshold")
	if len(text) == 0 {
		return nil, fmt.Errorf("select Threshold and Window from the lists")
	}
	dbLevel, err := strconv.Atoi(text)
	if err != nil {
		fmt.Printf("Conversion to int of 'threshold' error: %v\n", err)
		return nil, err
	}

	// construct the mlp
	mlp := MLP{
		epochs:       epochs,
		hiddenLayers: hiddenLayers,
		layerDepth:   hidLayersDepth,
		plot:         plot,
		learningRate: learningRate,
		samples:      make([]Sample, 0),
		momentum:     momentum,
		fftSize:      fftSize,
		fftWindow:    fftWindow,
		words:        make([]string, 0),
		wordWindow:   window,
		dbLevel:      dbLevel,
	}

	// retrieve the weights
	// first layer, one weight per line, (fftSize/2+1)*hiddenLayers
	mlp.link = make([][]Link, hiddenLayers+1)
	nwgts := (fftSize/2 + 1) * hidLayersDepth
	mlp.link[0] = make([]Link, nwgts)
	for i := 0; i < nwgts; i++ {
		scanner.Scan()
		line := scanner.Text()
		wgt, err := strconv.ParseFloat(line, 64)
		if err != nil {
			fmt.Printf("ParseFloat error: %v\n", err.Error())
			continue
		}
		mlp.link[0][i] = Link{wgt: wgt, wgtDelta: 0}
	}
	// Continue with remaining layers, one layer per line
	layer := 1
	for scanner.Scan() {
		line = scanner.Text()
		weights := strings.Split(line, ",")
		weights = weights[:len(weights)-1]
		mlp.link[layer] = make([]Link, len(weights))
		for i, wtStr := range weights {
			wt, err := strconv.ParseFloat(wtStr, 64)
			if err != nil {
				fmt.Printf("ParseFloat of %s error: %v", wtStr, err)
				continue
			}
			mlp.link[layer][i] = Link{wgt: wt, wgtDelta: 0}
		}
		layer++
	}
	if err = scanner.Err(); err != nil {
		fmt.Printf("scanner error: %s\n", err.Error())
		return nil, fmt.Errorf("scanner error: %v", err)
	}

	// construct node, init node[i][0].y to 1.0 (bias)
	mlp.node = make([][]Node, mlp.hiddenLayers+2)

	// input layer
	mlp.node[0] = make([]Node, fftSize/2+1)
	// set first node in the layer (bias) to 1
	mlp.node[0][0].y = 1.0

	// outer layer nodes
	olnodes := int(math.Ceil(math.Log2(float64(classes))))

	// output layer, which has no bias node
	mlp.node[mlp.hiddenLayers+1] = make([]Node, olnodes)

	// hidden layers
	for i := 1; i <= mlp.hiddenLayers; i++ {
		mlp.node[i] = make([]Node, mlp.layerDepth+1)
		// set first node in the layer (bias) to 1
		mlp.node[i][0].y = 1.0
	}

	// construct desired from classes, binary representation
	mlp.desired = make([]float64, olnodes)

	return &mlp, nil
}

// handleTesting performs pattern classification of the test data
func handleTestingMLP(w http.ResponseWriter, r *http.Request) {
	var (
		plot PlotT
		mlp  *MLP
		err  error
	)

	// Fill in the vocabulary
	if err = fillVocabulary(&plot); err != nil {
		fmt.Printf("fillVocabulary() error: %v\n", err)
		plot.Status = fmt.Sprintf("fillVocabulary error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTestingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Construct MLP instance containing MLP state
	mlp, err = newTestingMLP(r, &plot)
	if err != nil {
		fmt.Printf("newTestingMLP() error: %v\n", err)
		plot.Status = fmt.Sprintf("newTestingMLP() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTestingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Create the audio wav file, otherwise use what is already present
	newMsg := r.FormValue("message")
	if newMsg == "new" {
		fmedia, err := exec.LookPath("fmedia.exe")
		if err != nil {
			log.Fatal("fmedia is not available in PATH")
		} else {
			fmt.Printf("fmedia is available in path: %s\n", fmedia)
			cmd := exec.Command(fmedia, "--record", "-o", filepath.Join(dataDir, msgTestWav), "--until=10",
				"--format=int16", "--channels=mono", "--rate=8000", "-y", "--start-dblevel=-30", "--stop-dblevel=-30;1")
			stdoutStderr, err := cmd.CombinedOutput()
			if err != nil {
				fmt.Printf("stdout, stderr error from running fmedia: %v\n", err)
				plot.Status = fmt.Sprintf("stdout, stderr error from running fmedia: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplTestingMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			} else {
				fmt.Printf("fmedia output: %s\n", string(stdoutStderr))
			}
		}
	}

	// retrieve the classes used for training
	err = mlp.retrieveClasses()
	if err != nil {
		fmt.Printf("retrieveClasses error: %v\n", err)
		plot.Status = fmt.Sprintf("retrieveClasses error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTestingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// At end of all examples display TestingResults
	// Convert classification numbers to string in Results
	err = mlp.runClassification()
	if err != nil {
		fmt.Printf("runClassification() error: %v\n", err)
		plot.Status = fmt.Sprintf("runClassification() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTestingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Set default to time domain
	plot.TimeDomain = true
	// open and read the audio wav file
	// create wav decoder, audio IntBuffer, convert to audio FloatBuffer
	// loop over the FloatBuffer.Data and generate the Spectral Power Density
	// fill the grid with the PSD values
	// Option to plot time domain added.

	// Determine if time or frequency domain plot
	domain := r.FormValue("domain")
	// Time Domain
	if domain == "time" {
		plot.TimeDomain = true
	} else {
		plot.TimeDomain = false
	}

	if plot.TimeDomain {
		err := mlp.processTimeDomain(msgTestWav)
		if err != nil {
			fmt.Printf("processTimeDomain error: %v\n", err)
			plot.Status = fmt.Sprintf("processTimeDomain error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		plot.Status += fmt.Sprintf("Time Domain of %s plotted.", filepath.Join(dataDir, msgTestWav))
		// Frequency Domain
	} else {
		fftWindow := r.FormValue("fftwindow")

		txt := r.FormValue("fftsize")
		fftSize, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("fftsize int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("fftsize int conversion error: %s", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		err = mlp.processFrequencyDomain(msgTestWav, fftWindow, fftSize)
		if err != nil {
			fmt.Printf("processFrequencyDomain error: %v\n", err)
			plot.Status = fmt.Sprintf("processFrequencyDomain error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		plot.Status += fmt.Sprintf("Frequency Domain: PSD of %s plotted.", filepath.Join(dataDir, msgTestWav))
	}

	// Play the audio wav if fmedia is available in the PATH environment variable
	fmedia, err := exec.LookPath("fmedia.exe")
	if err != nil {
		log.Fatal("fmedia is not available in PATH")
	} else {
		fmt.Printf("fmedia is available in path: %s\n", fmedia)
		cmd := exec.Command(fmedia, filepath.Join(dataDir, msgTestWav))
		stdoutStderr, err := cmd.CombinedOutput()
		if err != nil {
			fmt.Printf("stdout, stderr error from running fmedia: %v\n", err)

		} else {
			fmt.Printf("fmedia output: %s\n", string(stdoutStderr))
		}
	}

	// Execute data on HTML template
	if err = tmplTestingMLP.Execute(w, mlp.plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}
}

// findEndpoints finds the minimum and maximum data values
func (ep *Endpoints) findEndpoints(input []float64) {
	ep.ymax = -math.MaxFloat64
	ep.ymin = math.MaxFloat64
	for _, y := range input {

		if y > ep.ymax {
			ep.ymax = y
		}
		if y < ep.ymin {
			ep.ymin = y
		}
	}
}

// processTimeDomain plots the time domain data from audio wav file
func (mlp *MLP) processTimeDomain(filename string) error {

	var (
		xscale    float64
		yscale    float64
		endpoints Endpoints
	)

	mlp.plot.Grid = make([]string, rows*cols)
	mlp.plot.Xlabel = make([]string, xlabels)
	mlp.plot.Ylabel = make([]string, ylabels)

	// Open the audio wav file
	f, err := os.Open(filepath.Join(dataDir, filename))
	if err == nil {
		defer f.Close()
		dec := wav.NewDecoder(f)
		bufInt := audio.IntBuffer{
			Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
			Data:   make([]int, maxSamples), SourceBitDepth: bitDepth}
		n, err := dec.PCMBuffer(&bufInt)
		if err != nil {
			fmt.Printf("PCMBuffer error: %v\n", err)
			return fmt.Errorf("PCMBuffer error: %v", err.Error())
		}
		bufFlt := bufInt.AsFloatBuffer()
		//fmt.Printf("%s samples = %d\n", filename, n)
		mlp.nsamples = n

		endpoints.findEndpoints(bufFlt.Data)
		// time starts at 0 and ends at #samples*sampling period
		endpoints.xmin = 0.0
		// #samples*sampling period, sampling period = 1/sampleRate
		endpoints.xmax = float64(mlp.nsamples) / float64(sampleRate)

		// EP means endpoints
		lenEPx := endpoints.xmax - endpoints.xmin
		lenEPy := endpoints.ymax - endpoints.ymin
		prevTime := 0.0
		prevAmpl := bufFlt.Data[0]

		// Calculate scale factors for x and y
		xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
		yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

		// This previous cell location (row,col) is on the line (visible)
		row := int((endpoints.ymax-bufFlt.Data[0])*yscale + .5)
		col := int((0.0-endpoints.xmin)*xscale + .5)
		mlp.plot.Grid[row*cols+col] = "online"

		// Store the amplitude in the plot Grid
		for n := 1; n < mlp.nsamples; n++ {
			// Current time
			currTime := float64(n) / float64(sampleRate)

			// This current cell location (row,col) is on the line (visible)
			row := int((endpoints.ymax-bufFlt.Data[n])*yscale + .5)
			col := int((currTime-endpoints.xmin)*xscale + .5)
			mlp.plot.Grid[row*cols+col] = "online"

			// Interpolate the points between previous point and current point;
			// draw a straight line between points.
			lenEdgeTime := math.Abs((currTime - prevTime))
			lenEdgeAmpl := math.Abs(bufFlt.Data[n] - prevAmpl)
			ncellsTime := int(float64(cols) * lenEdgeTime / lenEPx) // number of points to interpolate in x-dim
			ncellsAmpl := int(float64(rows) * lenEdgeAmpl / lenEPy) // number of points to interpolate in y-dim
			// Choose the biggest
			ncells := ncellsTime
			if ncellsAmpl > ncells {
				ncells = ncellsAmpl
			}

			stepTime := float64(currTime-prevTime) / float64(ncells)
			stepAmpl := float64(bufFlt.Data[n]-prevAmpl) / float64(ncells)

			// loop to draw the points
			interpTime := prevTime
			interpAmpl := prevAmpl
			for i := 0; i < ncells; i++ {
				row := int((endpoints.ymax-interpAmpl)*yscale + .5)
				col := int((interpTime-endpoints.xmin)*xscale + .5)
				// This cell location (row,col) is on the line (visible)
				mlp.plot.Grid[row*cols+col] = "online"
				interpTime += stepTime
				interpAmpl += stepAmpl
			}

			// Update the previous point with the current point
			prevTime = currTime
			prevAmpl = bufFlt.Data[n]

		}

		// Set plot status if no errors
		if len(mlp.plot.Status) == 0 {
			mlp.plot.Status = fmt.Sprintf("file %s plotted from (%.3f,%.3f) to (%.3f,%.3f)",
				filename, endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
		}

	} else {
		// Set plot status
		fmt.Printf("Error opening file %s: %v\n", filename, err)
		return fmt.Errorf("error opening file %s: %v", filename, err)
	}

	// Set plot status if no errors
	if len(mlp.plot.Status) == 0 {
		mlp.plot.Status = fmt.Sprintf("Status: Data plotted from (%.3f,%.3f) to (%.3f,%.3f)",
			endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / (xlabels - 1)
	x := endpoints.xmin
	// First label is empty for alignment purposes
	for i := range mlp.plot.Xlabel {
		mlp.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	for i := range mlp.plot.Ylabel {
		mlp.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	return nil
}

// Welch's Method and Bartlett's Method variation of the Periodogram
func (mlp *MLP) calculatePSD(audio []float64, PSD []float64, plottype, fftWindow string, fftSize int) (float64, float64, error) {
	// if used for creating examples, remove the mean of the |FFT|^2 and use the mag^2
	// for the FFT output, not the 10log10 dB.

	N := fftSize
	m := N / 2

	// map of window functions
	window := make(map[string]Window, len(winType))
	// Put the window functions in the map
	window["Bartlett"] = bartlett
	window["Welch"] = welch
	window["Hamming"] = hamming
	window["Hanning"] = hanning
	window["Rectangle"] = rectangle

	w, ok := window[fftWindow]
	if !ok {
		fmt.Printf("Invalid FFT window type: %v\n", mlp.fftWindow)
		return 0, 0, fmt.Errorf("invalid FFT window type: %v", mlp.fftWindow)
	}
	sumWindow := 0.0
	// sum the window values for PSD normalization due to windowing
	for i := 0; i < N; i++ {
		x := cmplx.Abs(w(i, N))
		sumWindow += x * x
	}

	psdMax := -math.MaxFloat64 // maximum PSD value
	psdMin := math.MaxFloat64  // minimum PSD value

	bufm := make([]complex128, m)
	bufN := make([]complex128, N)

	// part of K*Sum(w[i]*w[i]) PSD normalizer
	normalizerPSD := sumWindow

	// Initialize the PSD to zero as it is reused when creating examples
	for i := range PSD {
		PSD[i] = 0.0
	}

	// Find the mean by averaging the PSD sum
	psdSum := 0.0
	var sections int

	// Bartlett's method has no overlap of input data and uses the rectangle window
	if mlp.fftWindow == "Rectangle" {
		// full sections, account for partial section later
		sections = mlp.nsamples / N
		start := 0
		// Loop over sections and accumulate the PSD
		for i := 0; i < sections; i++ {

			for j := 0; j < N; j++ {
				bufN[j] = complex(audio[start+j], 0)
			}

			// Rectangle window, unity gain implicitly done

			// Perform N-point complex FFT and add squares to previous values in PSD
			// Normalize the PSD with the window sum, then convert to dB with 10*log10()
			fourierN := fft.FFT(bufN)
			x := cmplx.Abs(fourierN[0])
			PSD[0] += x * x
			for j := 1; j < m; j++ {
				// Use positive and negative frequencies -> bufN[N-j] = bufN[-j]
				xj := cmplx.Abs(fourierN[j])
				xNj := cmplx.Abs(fourierN[N-j])
				PSD[j] += xj*xj + xNj*xNj
				psdSum += PSD[j]
			}

			// part of K*Sum(w[i]*w[i]) PSD normalizer
			normalizerPSD += sumWindow
			// No overlap, skip to next N samples
			start += N
		}

		// left over samples if nsamples is not a multiple of FFT size
		diff := mlp.nsamples - start
		//fmt.Printf("left over samples = %d\n", diff)
		if diff > 0 {
			for j := 0; j < diff; j++ {
				bufN[j] = complex(audio[start+j], 0)
			}

			// zero-pad the remaining samples
			for i := diff; i < N; i++ {
				bufN[i] = 0
			}

			// Rectangle window, unity gain

			// Perform N-point complex FFT and add squares to previous values in PSD
			// Normalize the PSD with the window sum, then convert to dB with 10*log10()
			fourierN := fft.FFT(bufN)
			x := cmplx.Abs(fourierN[0])
			PSD[0] += x * x
			for j := 1; j < m; j++ {
				// Use positive and negative frequencies -> bufN[N-j] = bufN[-j]
				xj := cmplx.Abs(fourierN[j])
				xNj := cmplx.Abs(fourierN[N-j])
				PSD[j] += xj*xj + xNj*xNj
				psdSum += PSD[j]
			}

			// part of K*Sum(w[i]*w[i]) PSD normalizer
			normalizerPSD += sumWindow
		}

		// 50% overlap sections of audio input for non-rectangle windows, Welch's method
	} else {
		start := int(math.Min(float64(m), float64(mlp.nsamples)))
		// use two buffers, copy previous section to the front of current section
		for j := 0; j < start; j++ {
			bufm[j] = complex(audio[j], 0)
		}
		sections = (mlp.nsamples - m) / m
		for i := 0; i < sections; i++ {
			// copy previous section to front of current section
			copy(bufN, bufm)
			// Get the next fftSize/2 audio samples
			for j := 0; j < m; j++ {
				bufm[j] = complex(audio[start+j], 0)
			}
			// Put current section in back of previous
			copy(bufN[m:], bufm)

			// window the N samples with chosen window
			for k := 0; k < N; k++ {
				bufN[k] *= w(k, m)
			}

			// Perform N-point complex FFT and add squares to previous values in PSD
			// Normalize the PSD with the window sum, then convert to dB with 10*log10()
			fourierN := fft.FFT(bufN)
			x := cmplx.Abs(fourierN[0])
			PSD[0] += x * x
			for j := 1; j < m; j++ {
				// Use positive and negative frequencies -> bufN[N-j] = bufN[-j]
				xj := cmplx.Abs(fourierN[j])
				xNj := cmplx.Abs(fourierN[N-j])
				PSD[j] += xj*xj + xNj*xNj
				psdSum += PSD[j]
			}

			// part of K*Sum(w[i]*w[i]) PSD normalizer
			normalizerPSD += sumWindow

			start += m
		}

		// left over samples if nsamples is not a multiple of FFT size / 2
		diff := mlp.nsamples - start
		//fmt.Printf("left over samples = %d\n", diff)
		if diff > 0 {

			// copy previous section to front of current section
			copy(bufN, bufm)

			// Put the remaining samples in back of the previous m
			for j := 0; j < diff; j++ {
				bufN[m+j] = complex(audio[start+j], 0)
			}

			// zero-pad the remaining samples
			for i := m + diff; i < N; i++ {
				bufN[i] = 0
			}

			// window the N samples with chosen window
			for k := 0; k < m+diff; k++ {
				bufN[k] *= w(k, m)
			}

			// Perform N-point complex FFT and add squares to previous values in PSD
			// Normalize the PSD with the window sum, then convert to dB with 10*log10()
			fourierN := fft.FFT(bufN)
			x := cmplx.Abs(fourierN[0])
			PSD[0] += x * x
			for j := 1; j < m; j++ {
				// Use positive and negative frequencies -> bufN[N-j] = bufN[-j]
				xj := cmplx.Abs(fourierN[j])
				xNj := cmplx.Abs(fourierN[N-j])
				PSD[j] += xj*xj + xNj*xNj
				psdSum += PSD[j]
			}

			// part of K*Sum(w[i]*w[i]) PSD normalizer
			normalizerPSD += sumWindow
		}

	}

	// Normalize the PSD using K*Sum(w[i]*w[i])
	// Use log plot for wide dynamic range
	if plottype == "normalize" {
		// preprocess data for MLP by removing the mean
		psdMean := psdSum / float64(m*sections)
		for i := range PSD {
			PSD[i] = (PSD[i] - psdMean) / normalizerPSD
			if PSD[i] > psdMax {
				psdMax = PSD[i]
			}
			if PSD[i] < psdMin {
				psdMin = PSD[i]
			}
		}
		// Normalize to 1, otherwise the activation function saturates
		for i := range PSD {
			PSD[i] /= psdMax
		}
		// 10log10 in dB
	} else if plottype == "log" {
		for i := range PSD {
			PSD[i] /= normalizerPSD
			PSD[i] = 10.0 * math.Log10(PSD[i])
			if PSD[i] > psdMax {
				psdMax = PSD[i]
			}
			if PSD[i] < psdMin {
				psdMin = PSD[i]
			}
		}
	} else {
		return 0, 0, fmt.Errorf("calculatePSD invalid plot type: %s", plottype)
	}

	return psdMin, psdMax, nil
}

// processFrequencyDomain calculates the Power Spectral Density (PSD) and plots it
func (mlp *MLP) processFrequencyDomain(filename, fftWindow string, fftSize int) error {
	// Use complex128 for FFT computation
	// open and read the audio wav file
	// create wav decoder, audio IntBuffer, convert IntBuffer to audio FloatBuffer
	// loop over the FloatBuffer.Data and generate the FFT
	// fill the grid with the 10log10( mag^2 ) dB, Power Spectral Density

	var (
		endpoints Endpoints
		PSD       []float64 // power spectral density
		xscale    float64   // data to grid in x direction
		yscale    float64   // data to grid in y direction
	)

	mlp.plot.Grid = make([]string, rows*cols)
	mlp.plot.Xlabel = make([]string, xlabels)
	mlp.plot.Ylabel = make([]string, ylabels)

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD = make([]float64, fftSize/2)

	// Open the audio wav file
	f, err := os.Open(filepath.Join(dataDir, filename))
	if err == nil {
		defer f.Close()
		dec := wav.NewDecoder(f)
		bufInt := audio.IntBuffer{
			Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
			Data:   make([]int, maxSamples), SourceBitDepth: bitDepth}
		n, err := dec.PCMBuffer(&bufInt)
		if err != nil {
			fmt.Printf("PCMBuffer error: %v\n", err)
			return fmt.Errorf("PCMBuffer error: %v", err.Error())
		}
		bufFlt := bufInt.AsFloatBuffer()
		//fmt.Printf("%s samples = %d\n", filename, n)
		mlp.nsamples = n

		// calculate the PSD using Bartlett's or Welch's variant of the Periodogram
		psdMin, psdMax, err := mlp.calculatePSD(bufFlt.Data, PSD, "log", fftWindow, fftSize)
		if err != nil {
			fmt.Printf("calculatePSD error: %v\n", err)
			return fmt.Errorf("calculatePSD error: %v", err.Error())
		}

		endpoints.xmin = 0.0
		endpoints.xmax = float64(fftSize / 2) // equivalent to Nyquist critical frequency
		endpoints.ymin = psdMin
		endpoints.ymax = psdMax

		// EP means endpoints
		lenEPx := endpoints.xmax - endpoints.xmin
		lenEPy := endpoints.ymax - endpoints.ymin
		prevBin := 0.0
		prevPSD := PSD[0]

		// Calculate scale factors for x and y
		xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
		yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

		// This previous cell location (row,col) is on the line (visible)
		row := int((endpoints.ymax-PSD[0])*yscale + .5)
		col := int((0.0-endpoints.xmin)*xscale + .5)
		mlp.plot.Grid[row*cols+col] = "online"

		// Store the PSD in the plot Grid
		for bin := 1; bin < fftSize/2; bin++ {

			// This current cell location (row,col) is on the line (visible)
			row := int((endpoints.ymax-PSD[bin])*yscale + .5)
			col := int((float64(bin)-endpoints.xmin)*xscale + .5)
			mlp.plot.Grid[row*cols+col] = "online"

			// Interpolate the points between previous point and current point;
			// draw a straight line between points.
			lenEdgeBin := math.Abs((float64(bin) - prevBin))
			lenEdgePSD := math.Abs(PSD[bin] - prevPSD)
			ncellsBin := int(float64(cols) * lenEdgeBin / lenEPx) // number of points to interpolate in x-dim
			ncellsPSD := int(float64(rows) * lenEdgePSD / lenEPy) // number of points to interpolate in y-dim
			// Choose the biggest
			ncells := ncellsBin
			if ncellsPSD > ncells {
				ncells = ncellsPSD
			}

			stepBin := float64(float64(bin)-prevBin) / float64(ncells)
			stepPSD := float64(PSD[bin]-prevPSD) / float64(ncells)

			// loop to draw the points
			interpBin := prevBin
			interpPSD := prevPSD
			for i := 0; i < ncells; i++ {
				row := int((endpoints.ymax-interpPSD)*yscale + .5)
				col := int((interpBin-endpoints.xmin)*xscale + .5)
				// This cell location (row,col) is on the line (visible)
				mlp.plot.Grid[row*cols+col] = "online"
				interpBin += stepBin
				interpPSD += stepPSD
			}

			// Update the previous point with the current point
			prevBin = float64(bin)
			prevPSD = PSD[bin]

		}

		// Set plot status if no errors
		if len(mlp.plot.Status) == 0 {
			mlp.plot.Status = fmt.Sprintf("file %s plotted from (%.3f,%.3f) to (%.3f,%.3f)",
				filename, endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
		}

	} else {
		// Set plot status
		fmt.Printf("Error opening file %s: %v\n", filename, err)
		return fmt.Errorf("error opening file %s: %v", filename, err)
	}

	// Apply the  sampling rate in Hz to the x-axis using a scale factor
	// Convert the fft size to sampleRate/2, the Nyquist critical frequency
	sf := 0.5 * sampleRate / endpoints.xmax

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / (xlabels - 1)
	x := endpoints.xmin
	// First label is empty for alignment purposes
	for i := range mlp.plot.Xlabel {
		mlp.plot.Xlabel[i] = fmt.Sprintf("%.0f", x*sf)
		x += incr
	}

	// Construct the y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	for i := range mlp.plot.Ylabel {
		mlp.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	return nil
}

// Create a vocabulary for the message and display their values
func handleVocabularyGeneration(w http.ResponseWriter, r *http.Request) {
	var (
		plot PlotT
		mlp  *MLP
	)

	// Determine operation to perform on the vocabulary:  play, add, delete a word
	wordOp := r.FormValue("word")
	if wordOp == "play" {
		filename := r.FormValue("fileplaydelete")
		if len(filename) == 0 {
			fmt.Println("Enter filename for playing the word")
			plot.Status = "Enter filename for playing the word"
			// Write to HTTP using template and grid
			if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		audiowavdir := r.FormValue("audiowavdir")
		if len(audiowavdir) == 0 {
			fmt.Println("Enter directory for the audio wav file")
			plot.Status = "Enter director for the audio wav file"
			// Write to HTTP using template and grid
			if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Determine if time or frequency domain plot
		domain := r.FormValue("domain")
		// Time Domain
		if domain == "time" {
			plot.TimeDomain = true
		} else {
			plot.TimeDomain = false
		}

		if plot.TimeDomain {
			mlp = &MLP{plot: &plot}
			err := mlp.processTimeDomain(filepath.Join(audiowavdir, filename))
			if err != nil {
				fmt.Printf("processTimeDomain error: %v\n", err)
				plot.Status = fmt.Sprintf("processTimeDomain error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status += fmt.Sprintf("Time Domain of %s plotted.", filepath.Join(dataDir, audiowavdir, filename))
			// Frequency Domain
		} else {
			fftWindow := r.FormValue("fftwindow")

			txt := r.FormValue("fftsize")
			fftSize, err := strconv.Atoi(txt)
			if err != nil {
				fmt.Printf("fftsize int conversion error: %v\n", err)
				plot.Status = fmt.Sprintf("fftsize int conversion error: %s", err.Error())
				// Write to HTTP using template and grid
				if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			mlp = &MLP{plot: &plot, fftSize: fftSize, fftWindow: fftWindow}

			err = mlp.processFrequencyDomain(filepath.Join(audiowavdir, filename), fftWindow, fftSize)
			if err != nil {
				fmt.Printf("processFrequencyDomain error: %v\n", err)
				plot.Status = fmt.Sprintf("processFrequencyDomain error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status += fmt.Sprintf("Frequency Domain: PSD of %s plotted.", filepath.Join(dataDir, filename))
		}

		// Play the audio wav if fmedia is available in the PATH environment variable
		fmedia, err := exec.LookPath("fmedia.exe")
		if err != nil {
			log.Fatal("fmedia is not available in PATH")
		} else {
			fmt.Printf("fmedia is available in path: %s\n", fmedia)
			cmd := exec.Command(fmedia, filepath.Join(dataDir, audiowavdir, filename))
			stdoutStderr, err := cmd.CombinedOutput()
			if err != nil {
				fmt.Printf("stdout, stderr error from running fmedia: %v\n", err)

			} else {
				fmt.Printf("fmedia output: %s\n", string(stdoutStderr))
			}
		}
	} else if wordOp == "new" {
		mlp = &MLP{plot: &plot}
		filename := r.FormValue("filenew")
		if len(filename) == 0 {
			fmt.Println("Enter filename for the new word")
			plot.Status = "Enter filename for the new word"
			// Write to HTTP using template and grid
			if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		fmedia, err := exec.LookPath("fmedia.exe")
		if err != nil {
			log.Fatal("fmedia is not available in PATH")
		} else {
			fmt.Printf("fmedia is available in path: %s\n", fmedia)
			// filename includes the audiowav folder; eg.,  audiowavX/cat.wav, were X = 0, 1, 2, ...
			cmd := exec.Command(fmedia, "--record", "-o", filepath.Join(dataDir, filename), "--until=5",
				"--format=int16", "--channels=mono", "--rate=8000", "-y", "--start-dblevel=-30", "--stop-dblevel=-20;1")
			stdoutStderr, err := cmd.CombinedOutput()
			if err != nil {
				fmt.Printf("stdout, stderr error from running fmedia: %v\n", err)
				plot.Status = fmt.Sprintf("stdout, stderr error from running fmedia: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			} else {
				fmt.Printf("fmedia output: %s\n", string(stdoutStderr))
			}
		}
		// delete
	} else if wordOp == "delete" {
		mlp = &MLP{plot: &plot}
		filename := r.FormValue("fileplaydelete")
		if len(filename) == 0 {
			fmt.Println("Enter filename for deleting the word from the vocabulary")
			plot.Status = "Enter filename for deleting the word from the vocabulary"
			// Write to HTTP using template and grid
			if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		audiowavdir := r.FormValue("audiowavdir")
		if len(audiowavdir) == 0 {
			fmt.Println("Enter directory for the audio wav file")
			plot.Status = "Enter directory for the audio wav file"
			// Write to HTTP using template and grid
			if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		if filepath.Ext(filename) == ".wav" {
			if err := os.Remove(path.Join(dataDir, audiowavdir, filename)); err != nil {
				plot.Status = fmt.Sprintf("Remove %s error: %v", filename, err)
				// Write to HTTP using template and grid
				if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
		}
	} else {
		fmt.Println("Enter vocabulary parameters.")
		plot.Status = "Enter vocabulary parameters"
		// Write to HTTP using template and grid
		if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Execute data on HTML template
	if err := tmplVocabularyMLP.Execute(w, mlp.plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}

}

// executive creates the HTTP handlers, listens and serves
func main() {
	// Set up HTTP servers with handlers for training and testing the MLP Neural Network

	// Create HTTP handler for training
	http.HandleFunc(patternTrainingMLP, handleTrainingMLP)
	// Create HTTP handler for testing
	http.HandleFunc(patternTestingMLP, handleTestingMLP)
	// Create HTTP handler for vocabulary generation
	http.HandleFunc(patternVocabularyMLP, handleVocabularyGeneration)
	fmt.Printf("Multilayer Perceptron Neural Network Server listening on %v.\n", addr)
	http.ListenAndServe(addr, nil)
}
