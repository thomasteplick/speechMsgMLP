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

This application classifies audio wav files.  The spectrogram of
each file is calculated and the flattened spectrogram is the input to the MLP.
The MLP classifies the wav file based on its spectral content versus time. The test
results are shown.  The user can plot the time domain or the spectrogram
(frequency versus time) of the wav file.  The spectrogram is a three-dimentional
plot of the spectral power versus time.  The third dimension is a grayscale color.
Short-time Fourier Transforms (STFT) are used to compute the FFT from 32ms blocks
of audio data.
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
	fileSpectrogram      = "templates/spectrogram.html"   // html for speech time or spectrogram plots
	patternTrainingMLP   = "/speechMsgMLP"                // http handler for training the MLP
	patternTestingMLP    = "/speechMsgMLPtest"            // http handler for testing the MLP
	patternVocabularyMLP = "/speechMsgMLPvocabulary"      // http handler for adding words to the vocabulary
	patternSpectrogram   = "/speechspectrogram"           // http handler for speech spectrogram
	xlabels              = 11                             // # labels on x axis
	ylabels              = 11                             // # labels on y axis
	fileweights          = "weights.csv"                  // mlp weights
	a                    = 1.7159                         // activation function const
	b                    = 2.0 / 3.0                      // activation function const
	K1                   = b / a
	K2                   = a * a
	dataDir              = "data/"       // directory for the weights and audio wav files
	classes              = 8             // number of audio wav files to classify
	rows                 = 300           // rows in canvas
	cols                 = 300           // columns in canvas
	sampleRate           = 8000          // Hz or samples/sec
	maxSamples           = 10000         // max audio wav samples = 1.158 sec * sampleRate
	twoPi                = 2.0 * math.Pi // 2Pi
	bitDepth             = 16            // audio wav encoder/decoder sample size
	ncolors              = 5             // number of grayscale colors in spectrogram
	msgTestWav           = "message.wav" // Test message/subject wav file
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
	TestResults  string   // classified test message
	FFTSize      string   // 8192, 4098, 2048, 1024
	FFTWindow    string   // Bartlett, Welch, Hamming, Hanning, Rectangle
	Domain       string   // plot time or spectrogra domain
	Vocabulary   []string // vocabulary for the message
	WordWindow   string   // vocabulary word size for start/stop determination
	Threshold    string   // dB level for start/stop determination
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
	name      string    // audio wav file name
	desired   int       // numerical class of the audio wav file
	grayscale []float64 // spectrogram grayscale
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
	nsamples     int            // number of audio wav samples
	mse          []float64      // mean square error in output layer per epoch used in Learning Curve
	epochs       int            // number of epochs
	learningRate float64        // learning rate parameter
	momentum     float64        // delta weight scale consta
	hiddenLayers int            // number of hidden layers
	desired      []float64      // desired output of the sample
	layerDepth   int            // hidden layer number of nodes
	words        []string       // classified words in test message
	wordWindow   int            // message word window to accumulate audio level
	dbLevel      int            // message word audio level to determine start
	domain       string         // time or spectrogram plot
	audioMean    float64        // audio samples mean
	audioMax     float64        // audio samples maximum
	grayscale    map[int]string // grayscale for spectrogram
	fftSize      int            // FFT size for spectrogram
	fftWindow    string         // FFT window
}

// Window function type
type Window func(n int, m int) complex128

// global variables for parse and execution of the html template
var (
	tmplTrainingMLP   *template.Template
	tmplTestingMLP    *template.Template
	tmplVocabularyMLP *template.Template
	tmplSpectrogram   *template.Template
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

func (mlp *MLP) propagateForward(samp *Sample) error {

	// Assign spectrogram to input layer, i=0 is the bias equal to one
	for i, val := range samp.grayscale {
		mlp.node[0][i+1].y = val
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

	// number of directories containing the training spectrogram files
	const ndirs int = 10
	// read order of the audio spectrogram file directories
	readOrder := make([]int, ndirs)
	for i := 0; i < ndirs; i++ {
		readOrder[i] = i
	}

	// Initialize the weights

	// input layer
	// initialize the wgt and wgtDelta randomly, zero mean, normalize by fan-in
	for i := range mlp.link[0] {
		mlp.link[0][i].wgt = 2.0 * (rand.ExpFloat64() - .5) / float64(maxSamples+1)
		mlp.link[0][i].wgtDelta = 2.0 * (rand.ExpFloat64() - .5) / float64(maxSamples+1)
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

	// Create spectrogram files from the audio wav files if they don't exist
	// Note:  To recreate spectrogram files, delete spectrogram0/ files
	spectrogramDir := filepath.Join(dataDir, "spectrogram0")
	files, err := os.ReadDir(spectrogramDir)
	if err != nil {
		fmt.Printf("ReadDir for %s error: %v\n", spectrogramDir, err)
		return fmt.Errorf("ReadDir for %s error %v", spectrogramDir, err.Error())
	}
	if len(files) == 0 {
		err = mlp.createSpectrograms()
		if err != nil {
			fmt.Printf("createSpectrograms error: %v\n", err.Error())
			return fmt.Errorf("createSpectrograms error: %v", err)
		}
	}

	for n := 0; n < mlp.epochs; n++ {
		// loop over the spectrogram file directories
		// shuffle the spectrogram file directories read order
		rand.Shuffle(len(readOrder), func(i, j int) {
			readOrder[i], readOrder[j] = readOrder[j], readOrder[i]
		})

		// choose the spectrogram directory
		for _, val := range readOrder {
			spectrogramDir := filepath.Join(dataDir, fmt.Sprintf("spectrogram%d", val))
			if err := mlp.createExamples(spectrogramDir); err != nil {
				fmt.Printf("createExamples error: %v\n", err)
				return fmt.Errorf("createExamples error: %v", err.Error())
			}

			// Shuffle training examples
			rand.Shuffle(len(mlp.samples), func(i, j int) {
				mlp.samples[i], mlp.samples[j] = mlp.samples[j], mlp.samples[i]
			})

			// Loop over the training examples in chosen directory
			for _, samp := range mlp.samples {
				// Forward Propagation
				err := mlp.propagateForward(&samp)
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
	tmplSpectrogram = template.Must(template.ParseFiles(fileSpectrogram))
}

// createSpectrograms creates spectrogram csv files from the audio WAV files
func (mlp *MLP) createSpectrograms() error {
	// number of directories containing the training audio wav files
	const ndirs int = 10

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD := make([]float64, mlp.fftSize/2)
	data := make([]int, 2*maxSamples)
	// The number of Short-Time Fourier Transforms that will be done
	maxSTFTs := (maxSamples-mlp.fftSize)/(mlp.fftSize/2) + 1

	for i := 0; i < ndirs; i++ {
		audiowavDir := filepath.Join(dataDir, fmt.Sprintf("audiowav%d", i))
		specDir := filepath.Join(dataDir, fmt.Sprintf("spectrogram%d", i))

		// read in audio wav files and convert 16-bit samples to []float64
		// Each audio WAV file is a separate class
		files, err := os.ReadDir(audiowavDir)
		if err != nil {
			fmt.Printf("ReadDir for %s error: %v\n", audiowavDir, err)
			return fmt.Errorf("ReadDir for %s error %v", audiowavDir, err.Error())
		}
		class := 0
		for _, dirEntry := range files {
			name := dirEntry.Name()
			if filepath.Ext(name) == ".wav" {
				faudio, err := os.Open(filepath.Join(audiowavDir, name))
				if err != nil {
					fmt.Printf("Open %s error: %v\n", name, err)
					return fmt.Errorf("file Open %s error: %v", name, err.Error())
				}
				defer faudio.Close()
				// only process classes files
				if class == classes {
					return fmt.Errorf("can only process %v wav files", classes)
				}

				dec := wav.NewDecoder(faudio)
				bufInt := audio.IntBuffer{
					Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
					Data:   data, SourceBitDepth: bitDepth}
				n, err := dec.PCMBuffer(&bufInt)
				if err != nil {
					fmt.Printf("PCMBuffer error: %v\n", err)
					return fmt.Errorf("PCMBuffer error: %v", err.Error())
				}
				bufFlt := bufInt.AsFloatBuffer()
				//fmt.Printf("%s samples = %d\n", name, n)
				mlp.nsamples = n

				// loop over fltBuf and find the speech bounds
				bounds, err := mlp.findWords(bufFlt.Data)
				if err != nil {
					fmt.Printf("findWords error: %v", err)
					return fmt.Errorf("findWords error: %s", err.Error())
				}

				// Remove audio that doesn't contain speech; ie, noise
				nsamples := bounds[len(bounds)-1].stop - bounds[0].start
				if nsamples > 2*maxSamples {
					//fmt.Printf("%d audio samples is greater than max of %d\n", mlp.nsamples, maxSamples)
					nsamples = 2 * maxSamples
				}

				// Save the piece of the audio slice that contains speech.
				// Move the speech to front of buffer.
				for i := 0; i < nsamples; i++ {
					bufFlt.Data[i] = bufFlt.Data[bounds[0].start+i]
				}

				// normalize the audio to (-1, 1), remove the mean
				if err := mlp.normalizeAudio(bufFlt.Data[:nsamples], nsamples); err != nil {
					fmt.Printf("normalizeAudio error: %s\n", err.Error())
					return fmt.Errorf("normalizeAudio error: %v", err)
				}

				// Save STFTs to the csv spectrogram file
				fspec, err := os.Create(filepath.Join(specDir, strings.Replace(name, "wav", "csv", 1)))
				if err != nil {
					fmt.Printf("file Create %s error: %v\n", name, err)
					return fmt.Errorf("file Create %s error: %v", name, err.Error())
				}
				defer fspec.Close()

				// Create the spectrogram and save to disk
				// loop over the word boundaries using 50% overlap
				nSTFTs := 0
				for _, bound := range bounds {
					// loop over the frequency bins
					for smpl := bound.start; smpl < bound.stop; smpl += mlp.fftSize / 2 {
						// Skip noise, only speech is useful
						if mlp.inBoundsSample(smpl, bounds) {
							_, psdMax, err := mlp.calculatePSD(bufFlt.Data[smpl:smpl+mlp.fftSize], PSD, mlp.fftWindow, mlp.fftSize)
							if err != nil {
								fmt.Printf("calculatePSD error: %v\n", err)
								return fmt.Errorf("calculatePSD error: %v", err.Error())
							}
							// Write each STFT PSD to one row in the csv file
							n := 0
							for bin := 0; bin < mlp.fftSize/2-1; bin++ {
								// Relative power in the bin determines the grayscale color
								r := PSD[bin] / psdMax
								// five-color grayscale, 0 is white, 4 is black
								if r < .1 {
									n = 0
								} else if r < .25 {
									n = 1
								} else if r < .5 {
									n = 2
								} else if r < .8 {
									n = 3
								} else {
									n = 4
								}
								fmt.Fprintf(fspec, "%d,", n)
							}
							// no comma, insert newline
							r := PSD[mlp.fftSize/2-1] / psdMax
							// five-color grayscale, 0 is white, 4 is black
							if r < .1 {
								n = 0
							} else if r < .25 {
								n = 1
							} else if r < .5 {
								n = 2
							} else if r < .8 {
								n = 3
							} else {
								n = 4
							}
							fmt.Fprintf(fspec, "%d\n", n)
							nSTFTs++
						}
					}
				}
				// remaining STFT PSDs are empty
				for nSTFTs < maxSTFTs {
					for bin := 0; bin < mlp.fftSize/2-1; bin++ {
						fmt.Fprintf(fspec, "0,")
					}
					// no comma, insert newline
					fmt.Fprintf(fspec, "0\n")
					nSTFTs++
				}
				class++
			}
		}
	}

	return nil
}

// createExamples creates a slice of training examples
func (mlp *MLP) createExamples(spectrogramDir string) error {

	files, err := os.ReadDir(spectrogramDir)
	if err != nil {
		fmt.Printf("ReadDir for %s error: %v\n", spectrogramDir, err)
		return fmt.Errorf("ReadDir for %s error %v", spectrogramDir, err.Error())
	}
	// Each csv spectrogram file is a separate class
	class := 0
	for _, dirEntry := range files {
		name := dirEntry.Name()
		if filepath.Ext(name) == ".csv" {
			f, err := os.Open(path.Join(spectrogramDir, name))
			if err != nil {
				fmt.Printf("Open %s error: %v\n", name, err)
				return fmt.Errorf("file Open %s error: %v", name, err.Error())
			}
			defer f.Close()
			// only process classes files
			if class == classes {
				return fmt.Errorf("can only process %v spectrogram files", classes)
			}

			scanner := bufio.NewScanner(f)
			// get the STFT grayscale values, each row is a STFT of the PSD for the audio
			// serialize the 3-D spectrogram into a 1-D slice
			i := 0
			for scanner.Scan() {
				line := scanner.Text()
				items := strings.Split(line, ",")
				for _, str := range items {
					val, err := strconv.Atoi(str)
					if err != nil {
						fmt.Printf("String to int conversion error: %v\n", err.Error())
						return fmt.Errorf("string to int conversion error: %v", err)
					}
					mlp.samples[class].grayscale[i] = (float64(val) - 2.0) / 2.0
					i++
				}
			}

			if err = scanner.Err(); err != nil {
				fmt.Printf("scanner error: %s\n", err.Error())
				return fmt.Errorf("scanner error: %v", err)
			}

			// save the name of the audio wav without the ext
			mlp.samples[class].name = strings.Split(name, ".")[0]
			// The desired output of the MLP is class
			mlp.samples[class].desired = class

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

	txt = r.FormValue("wordwindow")
	if len(txt) == 0 {
		return nil, fmt.Errorf("select Threshold and Window from the lists")
	}
	window, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("Conversion to int of 'window' error: %v\n", err)
		return nil, err
	}

	txt = r.FormValue("threshold")
	if len(txt) == 0 {
		return nil, fmt.Errorf("select Threshold and Window from the lists")
	}
	dbLevel, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("Conversion to int of 'threshold' error: %v\n", err)
		return nil, err
	}

	fftWindow := r.FormValue("fftwindow")

	txt = r.FormValue("fftsize")
	fftSize, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("fftsize int conversion error: %v\n", err)
		return nil, err
	}

	mlp := MLP{
		hiddenLayers: hiddenLayers,
		layerDepth:   layerDepth,
		epochs:       epochs,
		learningRate: learningRate,
		momentum:     momentum,
		plot:         plot,
		Endpoints: Endpoints{
			ymin: math.MaxFloat64,
			ymax: -math.MaxFloat64,
			xmin: 0,
			xmax: float64(epochs - 1)},
		samples:    make([]Sample, classes),
		words:      make([]string, 0),
		wordWindow: window,
		dbLevel:    dbLevel,
		fftSize:    fftSize,
		fftWindow:  fftWindow,
	}
	// input layer nodes:  (#STFT)*fftSize/2+1, includes the bias node
	ilnodes := ((maxSamples-fftSize)/(fftSize/2)+1)*(fftSize/2) + 1

	// grayscale slice holds maxSamples-fftSize + 1 floats
	for i := range mlp.samples {
		mlp.samples[i].grayscale = make([]float64, ilnodes-1)
	}

	// construct link that holds the weights and weight deltas
	mlp.link = make([][]Link, hiddenLayers+1)

	// input layer
	mlp.link[0] = make([]Link, ilnodes*layerDepth)

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
	mlp.node[0] = make([]Node, ilnodes)
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
		// window, threshold, and weights to csv file, one layer per line
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
		fmt.Fprintf(f, "%d,%d,%d,%d,%f,%f,%d,%d,%d,%s\n",
			mlp.epochs, mlp.hiddenLayers, mlp.layerDepth, classes, mlp.learningRate,
			mlp.momentum, mlp.wordWindow, mlp.dbLevel, mlp.fftSize, mlp.fftWindow)
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

// Welch's Method and Bartlett's Method variation of the Periodogram
func (mlp *MLP) calculatePSD(audio []float64, PSD []float64, fftWindow string, fftSize int) (float64, float64, error) {

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
		fmt.Printf("Invalid FFT window type: %v\n", fftWindow)
		return 0, 0, fmt.Errorf("invalid FFT window type: %v", fftWindow)
	}

	bufN := make([]complex128, N)

	for j := 0; j < len(audio); j++ {
		bufN[j] = complex(audio[j], 0)
	}

	// zero-pad the remaining samples
	for i := len(audio); i < N; i++ {
		bufN[i] = 0
	}

	// window the N samples with chosen window
	for k := 0; k < N; k++ {
		bufN[k] *= w(k, m)
	}

	// Perform N-point complex FFT and add squares to previous values in PSD
	fourierN := fft.FFT(bufN)
	x := cmplx.Abs(fourierN[0])
	PSD[0] = x * x
	psdMax := PSD[0]
	psdAvg := PSD[0]
	for j := 1; j < m; j++ {
		// Use positive and negative frequencies -> bufN[N-j] = bufN[-j]
		xj := cmplx.Abs(fourierN[j])
		xNj := cmplx.Abs(fourierN[N-j])
		PSD[j] = xj*xj + xNj*xNj
		if PSD[j] > psdMax {
			psdMax = PSD[j]
		}
		psdAvg += PSD[j]
	}

	return psdAvg / float64(m), psdMax, nil
}

// findWords finds the word boundaries in the testing message
func (mlp *MLP) findWords(data []float64) ([]Bound, error) {

	// prevent oscillation about threshold
	const hystersis = 0.8

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
	// Minimum audio integration to determine when word begins and ends
	levelSum := float64(win) * avg
	buf := make([]float64, win)

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
				break
			}
			k++
		}
		j++
	}

	return bounds, nil
}

// normalizeAudio removes the mean and constrains the values to (-1,1)
func (mlp *MLP) normalizeAudio(audio []float64, nsamples int) error {

	// find the mean and remove it from audio
	sum := 0.0
	for i := 0; i < nsamples; i++ {
		sum += audio[i]
	}
	mean := sum / float64(nsamples)
	max := -math.MaxFloat64

	// remove the mean and find the maximum
	for i := 0; i < nsamples; i++ {
		audio[i] -= mean
		mag := math.Abs(audio[i])
		if mag > max {
			max = mag
		}
	}
	mlp.audioMean = mean
	mlp.audioMax = max

	// normalize the audio to (-1, 1)
	for i := 0; i < nsamples; i++ {
		audio[i] /= max
	}

	return nil
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
		Data:   make([]int, 2*maxSamples), SourceBitDepth: bitDepth}
	n, err := dec.PCMBuffer(&bufInt)
	if err != nil {
		fmt.Printf("PCMBuffer error: %v\n", err)
		return fmt.Errorf("PCMBuffer error: %v", err.Error())
	}
	bufFlt := bufInt.AsFloatBuffer()
	//fmt.Printf("%s samples = %d\n", filename, n)
	mlp.nsamples = n

	// loop over fltBuf and find the speech in the audio
	bounds, err := mlp.findWords(bufFlt.Data)
	if err != nil {
		fmt.Printf("findWords error: %v", err)
		return fmt.Errorf("findWords error: %s", err.Error())
	}

	// Find the bounds of the subject which may contain more than one word
	// Propagate forward and classify the subject
	// Insert the class in mlp.words using mlp.samples[class].name
	start := bounds[0].start
	stop := bounds[len(bounds)-1].stop
	fmt.Printf("start = %.3f, stop = %.3f\n", float64(start)*.000125, float64(stop)*.000125)
	mlp.nsamples = stop - start
	if mlp.nsamples > 2*maxSamples {
		//fmt.Printf("%d audio samples is greater than max of %d\n", mlp.nsamples, 2*maxSamples)
		mlp.nsamples = 2 * maxSamples
	}

	// copy the speech part of the audio
	for i := 0; i < mlp.nsamples; i++ {
		bufFlt.Data[i] = bufFlt.Data[start+i]
	}

	// normalize the audio to (-1, 1), remove the mean
	if err := mlp.normalizeAudio(bufFlt.Data, mlp.nsamples); err != nil {
		fmt.Printf("normalizeAudio error: %s\n", err.Error())
		return fmt.Errorf("normalizeAudio error: %v", err)
	}

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD := make([]float64, mlp.fftSize/2)

	// The number of Short-Time Fourier Transforms that will be done
	maxSTFTs := (maxSamples-mlp.fftSize)/(mlp.fftSize/2) + 1

	// serialize the 3-D spectrogram to 1-D slice for input layer of MLP
	grayscale := make([]float64, maxSTFTs*(mlp.fftSize/2))

	// Create the spectrogram
	// loop over the word boundaries using 50% overlap
	i := 0
	nSTFTs := 0
	for _, bound := range bounds {
		// loop over the frequency bins
		for smpl := bound.start; smpl < bound.stop; smpl += mlp.fftSize / 2 {
			// Skip noise, only speech is useful
			if mlp.inBoundsSample(smpl, bounds) {
				_, psdMax, err := mlp.calculatePSD(bufFlt.Data[smpl:smpl+mlp.fftSize], PSD, mlp.fftWindow, mlp.fftSize)
				if err != nil {
					fmt.Printf("calculatePSD error: %v\n", err)
					return fmt.Errorf("calculatePSD error: %v", err.Error())
				}
				// Write the spectrogram
				for bin := 0; bin < mlp.fftSize/2; bin++ {
					r := PSD[bin] / psdMax
					val := 0.0
					// five-color grayscale: -1.0, -.5, 0, .5, 1.0
					if r < .1 {
						val = -1.0
					} else if r < .25 {
						val = -0.5
					} else if r < .5 {
						val = 0
					} else if r < .8 {
						val = 0.5
					} else {
						val = 1.0
					}
					grayscale[i] = val
					i++
				}
				nSTFTs++
			}
		}
	}
	// remaining STFTs are empty
	for nSTFTs < maxSTFTs {
		for bin := 0; bin < mlp.fftSize/2; bin++ {
			grayscale[i] = -1.0
		}
		nSTFTs++
	}

	samp := Sample{desired: 0, grayscale: grayscale, name: ""}
	err = mlp.propagateForward(&samp)
	if err != nil {
		return fmt.Errorf("forward propagation error: %s", err.Error())
	}
	err = mlp.determineClass()
	if err != nil {
		return fmt.Errorf("determineClass error: %s", err.Error())
	}

	mlp.plot.TestResults = strings.Join(mlp.words, " ")

	mlp.plot.LearningRate = strconv.FormatFloat(mlp.learningRate, 'f', -1, 64)
	mlp.plot.Momentum = strconv.FormatFloat(mlp.momentum, 'f', -1, 64)
	mlp.plot.HiddenLayers = strconv.Itoa(mlp.hiddenLayers)
	mlp.plot.LayerDepth = strconv.Itoa(mlp.layerDepth)
	mlp.plot.Classes = strconv.Itoa(classes)
	mlp.plot.Epochs = strconv.Itoa(mlp.epochs)
	mlp.plot.WordWindow = strconv.Itoa(mlp.wordWindow)
	mlp.plot.Threshold = strconv.Itoa(mlp.dbLevel)
	mlp.plot.FFTSize = strconv.Itoa(mlp.fftSize)
	mlp.plot.FFTWindow = mlp.fftWindow
	if mlp.domain == "spectrogram" {
		mlp.plot.Domain = "Spectrogram (Hz/sec)"
	} else {
		mlp.plot.Domain = "Time Domain (sec)"
	}

	mlp.plot.Status = "Testing results completed."

	return nil
}

// newTestingMLP constructs an MLP from the saved mlp weights and parameters
func newTestingMLP(plot *PlotT) (*MLP, error) {
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
	if len(items) != 10 {
		fmt.Printf("Testing parameters missing, should be 10, is %d\n", len(items))
		return nil, fmt.Errorf("testing parameters missing, run Train first")
	}

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

	wordWindow, err := strconv.Atoi(items[6])
	if err != nil {
		fmt.Printf("Conversion to int of 'wordWindow' error: %v\n", err)
		return nil, err
	}

	dbLevel, err := strconv.Atoi(items[7])
	if err != nil {
		fmt.Printf("Conversion to int of 'threshold' error: %v\n", err)
		return nil, err
	}

	fftSize, err := strconv.Atoi(items[8])
	if err != nil {
		fmt.Printf("Conversion to int of 'fftSize' error: %v\n", err)
		return nil, err
	}

	fftWindow := items[9]

	// construct the mlp
	mlp := MLP{
		epochs:       epochs,
		hiddenLayers: hiddenLayers,
		layerDepth:   hidLayersDepth,
		plot:         plot,
		learningRate: learningRate,
		samples:      make([]Sample, 0),
		momentum:     momentum,
		words:        make([]string, 0),
		wordWindow:   wordWindow,
		dbLevel:      dbLevel,
		fftSize:      fftSize,
		fftWindow:    fftWindow,
	}

	// retrieve the weights
	// first layer, one weight per line, (transformSize+1)*hiddenLayers
	mlp.link = make([][]Link, hiddenLayers+1)

	ilnodes := ((maxSamples-fftSize)/(fftSize/2)+1)*(fftSize/2) + 1

	nwgts := ilnodes * hidLayersDepth
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
	mlp.node[0] = make([]Node, ilnodes)
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

// handleTestingMLP performs pattern classification of the test data
func handleTestingMLP(w http.ResponseWriter, r *http.Request) {
	// open and read the audio wav file
	// create wav decoder, audio IntBuffer, convert to audio FloatBuffer
	// loop over the Float Buffer data and generate the spectrogram
	// fill the grid with the values
	// Option to plot time domain added.
	// Option to plot spectrogram output added.

	var (
		plot      PlotT
		mlp       *MLP
		err       error
		wordsOnly bool = false
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
	mlp, err = newTestingMLP(&plot)
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
			cmd := exec.Command(fmedia, "--record", "-o", filepath.Join(dataDir, msgTestWav), "--until=5",
				"--format=int16", "--channels=mono", "--rate=8000", "-y", "--start-dblevel=-70", "--stop-dblevel=-30;1")
			stdoutStderr, err := cmd.CombinedOutput()
			if err != nil {
				fmt.Printf("stdout, stderr error from running fmedia: %v\n", err.Error())
				plot.Status = fmt.Sprintf("stdout, stderr error from running fmedia: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplTestingMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v", err)
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

	// Determine if time or spectrogram domain plot
	mlp.domain = r.FormValue("domain")
	if len(mlp.domain) == 0 {
		mlp.domain = "time"
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

	if mlp.domain == "spectrogram" {
		if len(r.FormValue("wordsonly")) > 0 {
			wordsOnly = true
		}
		mlp.grayscale = make(map[int]string)
		for i := 0; i < ncolors; i++ {
			mlp.grayscale[i] = fmt.Sprintf("gs%d", i)
		}

		err = mlp.processSpectrogram(msgTestWav, mlp.fftWindow, wordsOnly, mlp.fftSize)
		if err != nil {
			fmt.Printf("proessSpectrogram error: %v\n", err)
			plot.Status = fmt.Sprintf("processSpectrogram error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		plot.Status = "Spectrogram plotted."
	} else {
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
		plot.Status = fmt.Sprintf("Time Domain of %s plotted.", filepath.Join(dataDir, msgTestWav))
	}

	// Play the audio wav if fmedia is available in the PATH environment variable
	fmedia, err := exec.LookPath("fmedia.exe")
	if err != nil {
		log.Fatal("fmedia is not available in PATH")
	} else {
		fmt.Printf("fmedia is available in path: %s\n", fmedia)
		file := msgTestWav
		cmd := exec.Command(fmedia, filepath.Join(dataDir, file))
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
			Data:   make([]int, 2*maxSamples), SourceBitDepth: bitDepth}
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

// Create a vocabulary for the message and display their values
func handleVocabularyGeneration(w http.ResponseWriter, r *http.Request) {
	var (
		plot       PlotT
		mlp        *MLP
		wordWindow int  = 0
		wordsOnly  bool = false
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
		if domain == "spectrogram" {
			plot.Domain = "Spectrogram (Hz/sec)"
		} else {
			plot.Domain = "Time Domain (sec)"
		}

		if domain == "time" {
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
			// Spectrogram
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
			if len(r.FormValue("wordsonly")) > 0 {
				wordsOnly = true
				txt := r.FormValue("wordwindow")
				if len(txt) == 0 {
					fmt.Println("Word window not defined for spectrogram  domain")
					plot.Status = "Word window not defined for spectrogram  domain"
					// Write to HTTP using template and grid
					if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
						log.Fatalf("Write to HTTP output using template with error: %v\n", err)
					}
					return
				}
				wordWindow, err = strconv.Atoi(txt)
				if err != nil {
					fmt.Printf("Conversion to int for 'wordwindow' error: %v\n", err)
					plot.Status = "Conversion to int for 'window' error"
					// Write to HTTP using template and grid
					if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
						log.Fatalf("Write to HTTP output using template with error: %v\n", err)
					}
					return
				}
			}

			mlp = &MLP{plot: &plot, wordWindow: wordWindow, fftSize: fftSize}
			mlp.grayscale = make(map[int]string)
			for i := 0; i < ncolors; i++ {
				mlp.grayscale[i] = fmt.Sprintf("gs%d", i)
			}

			err = mlp.processSpectrogram(filepath.Join(audiowavdir, filename), fftWindow, wordsOnly, fftSize)
			if err != nil {
				fmt.Printf("processSpectrogram error: %v\n", err)
				plot.Status = fmt.Sprintf("processSpectrogram error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplVocabularyMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status += fmt.Sprintf("Spectrogram of %s plotted.", filepath.Join(dataDir, audiowavdir, filename))
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
				"--format=int16", "--channels=mono", "--rate=8000", "-y", "--start-dblevel=-70", "--stop-dblevel=-20;1")
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

// inBoundsSample checks if the sample is inside word boundaries
func (mlp *MLP) inBoundsSample(smpl int, bounds []Bound) bool {
	margin := mlp.fftSize / 2
	for _, bound := range bounds {
		if smpl > (bound.start-margin) && smpl < (bound.stop-margin) {
			return true
		}
	}
	return false
}

// processSpectrogram creates a spectrogram of the speech waveform
func (mlp *MLP) processSpectrogram(filename, fftWindow string, wordsOnly bool, fftSize int) error {

	// get audio samples from audio wav file
	// open and read the audio wav file
	// create wav decoder, audio IntBuffer, convert IntBuffer to audio FloatBuffer
	var (
		endpoints Endpoints
		PSD       []float64 // power spectral density
		xscale    float64   // data to grid in x direction
		yscale    float64   // data to grid in y direction
		bounds    []Bound   // word boundaries in the audio
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
			Data:   make([]int, 2*maxSamples), SourceBitDepth: bitDepth}
		n, err := dec.PCMBuffer(&bufInt)
		if err != nil {
			fmt.Printf("PCMBuffer error: %v\n", err)
			return fmt.Errorf("PCMBuffer error: %v", err.Error())
		}
		bufFlt := bufInt.AsFloatBuffer()
		//fmt.Printf("%s samples = %d\n", filename, n)
		mlp.nsamples = n
		// x-axis is time or sample, y-axis is frequency
		endpoints.xmin = 0.0
		endpoints.xmax = float64(mlp.nsamples)
		endpoints.ymin = 0.0
		endpoints.ymax = float64(fftSize / 2) // equivalent to Nyquist critical frequency

		// Calculate scale factors to convert physical units to screen units
		xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
		yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

		// number of cells to interpolate in time and frequency
		// round up so the cells in the plot grid are connected
		ncellst := int((math.Ceil(float64(cols) * float64(fftSize/2) / float64(mlp.nsamples))))
		ncellsf := int(math.Ceil(float64(rows) / float64((fftSize / 2))))

		stepTime := float64((fftSize / 2) / ncellst)
		stepFreq := 1.0 / float64(ncellsf)

		// if wordsOnly, only do calculatePSD for samples inside the word boundaries to minimize
		// checking the spectrum of noise.  This would give a broad range of frequencies which
		// is not of interest.
		if wordsOnly {
			// loop over fltBuf and find the speech bounds
			bounds, err = mlp.findWords(bufFlt.Data)
			if err != nil {
				fmt.Printf("findWords error: %v", err)
				return fmt.Errorf("findWords error: %s", err.Error())
			}
		}

		// for loop over samples, increment by fftSize/2, calculatePSD on the batch
		// Overlap by 50% due to non-rectangular window to avoid Gibbs phenomenon
		for smpl := 0; smpl < mlp.nsamples; smpl += fftSize / 2 {
			if !wordsOnly || mlp.inBoundsSample(smpl, bounds) {
				// calculate the PSD using Bartlett's or Welch's variant of the Periodogram
				end := smpl + fftSize
				if end > mlp.nsamples {
					end = mlp.nsamples
				}
				_, psdMax, err := mlp.calculatePSD(bufFlt.Data[smpl:end], PSD, fftWindow, fftSize)
				if err != nil {
					fmt.Printf("calculatePSD error: %v\n", err)
					return fmt.Errorf("calculatePSD error: %v", err.Error())
				}

				// for loop over the frequency bins in the PSD
				for bin := 0; bin < fftSize/2; bin++ {
					// find the grayscale color based on bin power
					// largest power is black, smallest power is white
					// shades of gray in-between black and white
					var gs string
					r := PSD[bin] / psdMax
					if r < .1 {
						gs = mlp.grayscale[4]
					} else if r < .25 {
						gs = mlp.grayscale[3]
					} else if r < .5 {
						gs = mlp.grayscale[2]
					} else if r < .8 {
						gs = mlp.grayscale[1]
					} else {
						gs = mlp.grayscale[0]
					}

					// interpolate in time
					interpTime := float64(smpl)
					for nct := 0; nct < ncellst; nct++ {
						col := int((interpTime-endpoints.xmin)*xscale + .5)
						if col >= cols {
							col = cols - 1
						}
						// interpolate in frequency
						interpFreq := float64(bin)
						for ncf := 0; ncf < ncellsf; ncf++ {
							row := int((endpoints.ymax-interpFreq)*yscale + .5)
							if row < 0 {
								row = 0
							}
							// Store the color in the plot Grid
							mlp.plot.Grid[row*cols+col] = gs
							interpFreq += stepFreq
						}
						interpTime += stepTime
					}
				}
			}
		}
	} else {
		// Set plot status
		fmt.Printf("Error opening file %s: %v\n", filename, err)
		return fmt.Errorf("error opening file %s: %v", filename, err)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / ((xlabels - 1) * sampleRate)
	x := endpoints.xmin / sampleRate
	// First label is empty for alignment purposes
	for i := range mlp.plot.Xlabel {
		mlp.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Apply the  sampling rate in Hz to the y-axis using a scale factor
	// Convert the fft size to sampleRate/2, the Nyquist critical frequency
	sf := 0.5 * sampleRate / endpoints.ymax

	// Construct y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	// First label is empty for alignment purposes
	for i := range mlp.plot.Ylabel {
		mlp.plot.Ylabel[i] = fmt.Sprintf("%.0f", y*sf)
		y += incr
	}

	return nil
}

// Create a spectrogram of the speech waveform
func handleSpectrogram(w http.ResponseWriter, r *http.Request) {
	var (
		plot       PlotT
		mlp        *MLP
		wordWindow int  = 0
		wordsOnly  bool = false
	)

	// Determine operation to perform on the speech waveform:  play, add, delete
	wordOp := r.FormValue("wordop")
	if wordOp == "play" {
		filename := r.FormValue("fileplaydelete")
		if len(filename) == 0 {
			fmt.Println("Enter filename for playing the word")
			plot.Status = "Enter filename for playing the word"
			// Write to HTTP using template and grid
			if err := tmplSpectrogram.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		audiowavdir := r.FormValue("audiowavdir")
		if len(audiowavdir) == 0 {
			fmt.Println("Enter directory for the audio wav file")
			plot.Status = "Enter director for the audio wav file"
			// Write to HTTP using template and grid
			if err := tmplSpectrogram.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Determine if time or spectrogram domain plot
		domain := r.FormValue("domain")
		if domain == "spectrogram" {
			plot.Domain = "Spectrogram (Hz/sec)"
		} else {
			plot.Domain = "Time Domain (sec)"
		}

		if domain == "time" {
			mlp = &MLP{plot: &plot}
			err := mlp.processTimeDomain(filepath.Join(audiowavdir, filename))
			if err != nil {
				fmt.Printf("processTimeDomain error: %v\n", err)
				plot.Status = fmt.Sprintf("processTimeDomain error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplSpectrogram.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status += fmt.Sprintf("Time Domain of %s plotted.", filepath.Join(dataDir, audiowavdir, filename))
			// Spectrogram Domain
		} else {
			fftWindow := r.FormValue("fftwindow")

			txt := r.FormValue("fftsize")
			fftSize, err := strconv.Atoi(txt)
			if err != nil {
				fmt.Printf("fftsize int conversion error: %v\n", err)
				plot.Status = fmt.Sprintf("fftsize int conversion error: %s", err.Error())
				// Write to HTTP using template and grid
				if err := tmplSpectrogram.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}

			if len(r.FormValue("wordsonly")) > 0 {
				wordsOnly = true
				txt := r.FormValue("wordwindow")
				if len(txt) == 0 {
					fmt.Println("Word window not defined for spectrogram  domain")
					plot.Status = "Word window not defined for spectrogram  domain"
					// Write to HTTP using template and grid
					if err := tmplSpectrogram.Execute(w, plot); err != nil {
						log.Fatalf("Write to HTTP output using template with error: %v\n", err)
					}
					return
				}
				wordWindow, err = strconv.Atoi(txt)
				if err != nil {
					fmt.Printf("Conversion to int for 'wordwindow' error: %v\n", err)
					plot.Status = "Conversion to int for 'window' error"
					// Write to HTTP using template and grid
					if err := tmplSpectrogram.Execute(w, plot); err != nil {
						log.Fatalf("Write to HTTP output using template with error: %v\n", err)
					}
					return
				}
			}

			mlp = &MLP{plot: &plot, wordWindow: wordWindow, fftSize: fftSize}
			mlp.grayscale = make(map[int]string)
			for i := 0; i < ncolors; i++ {
				mlp.grayscale[i] = fmt.Sprintf("gs%d", i)
			}

			err = mlp.processSpectrogram(filepath.Join(audiowavdir, filename), fftWindow, wordsOnly, fftSize)
			if err != nil {
				fmt.Printf("processSpectrogram error: %v\n", err)
				plot.Status = fmt.Sprintf("processSpectrogram error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplSpectrogram.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status += fmt.Sprintf("Spectrogram of %s plotted.", filepath.Join(dataDir, audiowavdir, filename))
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
			if err := tmplSpectrogram.Execute(w, plot); err != nil {
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
				"--format=int16", "--channels=mono", "--rate=8000", "-y", "--start-dblevel=-70", "--stop-dblevel=-20;1")
			stdoutStderr, err := cmd.CombinedOutput()
			if err != nil {
				fmt.Printf("stdout, stderr error from running fmedia: %v\n", err)
				plot.Status = fmt.Sprintf("stdout, stderr error from running fmedia: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplSpectrogram.Execute(w, plot); err != nil {
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
			if err := tmplSpectrogram.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		audiowavdir := r.FormValue("audiowavdir")
		if len(audiowavdir) == 0 {
			fmt.Println("Enter directory for the audio wav file")
			plot.Status = "Enter directory for the audio wav file"
			// Write to HTTP using template and grid
			if err := tmplSpectrogram.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		if filepath.Ext(filename) == ".wav" {
			if err := os.Remove(path.Join(dataDir, audiowavdir, filename)); err != nil {
				plot.Status = fmt.Sprintf("Remove %s error: %v", filename, err)
				// Write to HTTP using template and grid
				if err := tmplSpectrogram.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
		}
	} else {
		fmt.Println("Enter spectrogram parameters.")
		plot.Status = "Enter spectrogram parameters"
		// Write to HTTP using template and grid
		if err := tmplSpectrogram.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Execute data on HTML template
	if err := tmplSpectrogram.Execute(w, mlp.plot); err != nil {
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
	// Create HTTP handler for spectrogram generation
	http.HandleFunc(patternSpectrogram, handleSpectrogram)
	fmt.Printf("Multilayer Perceptron Neural Network Server listening on %v.\n", addr)
	http.ListenAndServe(addr, nil)
}
