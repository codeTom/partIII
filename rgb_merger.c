// Reads movies and merge frames to rgb tiffs
// Based on movie2tiff.c by Jurij Kotar
// Compile with gcc -o rgb_merge rgb_merger.c -O3 -Wall -ltiff

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <tiffio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <stdbool.h>
#include <libgen.h>
#include <glob.h>


//
// Common camera defines
//
#define CAMERA_MOVIE_MAGIC 0x496d6554 // TemI
#define CAMERA_MOVIE_VERSION 1
#define CAMERA_TYPE_IIDC	1
#define CAMERA_TYPE_ANDOR	2
#define CAMERA_TYPE_XIMEA	3
#define CAMERA_PIXELMODE_MONO_8		1
#define CAMERA_PIXELMODE_MONO_16BE	2 // Big endian
#define CAMERA_PIXELMODE_MONO_16LE	3 // Little endian

//
// IIDC defines
//
#define IIDC_MOVIE_HEADER_LENGTH 172
// Feature modes
#define IIDC_FEATURE_MODE_OFF ( 1<<0 )
#define IIDC_FEATURE_MODE_RELATIVE ( 1<<1 )
#define IIDC_FEATURE_MODE_ABSOLUTE ( 1<<2 )
#define IIDC_FEATURE_MODE_AUTO ( 1<<3 )
#define IIDC_FEATURE_MODE_ONEPUSH ( 1<<4 )
#define IIDC_FEATURE_MODE_ADVANCED ( 1<<5 )
// Trigger
#define IIDC_TRIGGER_INTERNAL  -1
#define IIDC_TRIGGER_EXTERNAL0 0
#define IIDC_TRIGGER_EXTERNAL1 1
#define IIDC_TRIGGER_EXTERNAL15 7

//
// Andor defines
//
#define ANDOR_MOVIE_HEADER_LENGTH 128
// VS Speeds
#define ANDOR_VALUE_VS_SPEED_MIN 4
#define ANDOR_VALUE_VS_SPEED_MAX 0
#define ANDOR_VALUE_VS_SPEED_0_3 0
#define ANDOR_VALUE_VS_SPEED_0_5 1
#define ANDOR_VALUE_VS_SPEED_0_9 2
#define ANDOR_VALUE_VS_SPEED_1_7 3
#define ANDOR_VALUE_VS_SPEED_3_3 4
// VS Amplitudes
#define ANDOR_VALUE_VS_AMPLITUDE_MIN 0
#define ANDOR_VALUE_VS_AMPLITUDE_MAX 4
#define ANDOR_VALUE_VS_AMPLITUDE_0 0
#define ANDOR_VALUE_VS_AMPLITUDE_1 1
#define ANDOR_VALUE_VS_AMPLITUDE_2 2
#define ANDOR_VALUE_VS_AMPLITUDE_3 3
#define ANDOR_VALUE_VS_AMPLITUDE_4 4
// Shutter
#define ANDOR_VALUE_SHUTTER_AUTO 0
#define ANDOR_VALUE_SHUTTER_OPEN 1
#define ANDOR_VALUE_SHUTTER_CLOSE 2
// Cooler
#define ANDOR_VALUE_COOLER_OFF 0
#define ANDOR_VALUE_COOLER_ON 1
// Cooler mode
#define ANDOR_VALUE_COOLER_MODE_RETURN 0
#define ANDOR_VALUE_COOLER_MODE_MAINTAIN 1
// Fan
#define ANDOR_VALUE_FAN_FULL 0
#define ANDOR_VALUE_FAN_LOW 1
#define ANDOR_VALUE_FAN_OFF 2
// ADC
#define ANDOR_VALUE_ADC_14BIT 0
#define ANDOR_VALUE_ADC_16BIT 1
// Amplifier
#define ANDOR_VALUE_AMPLIFIER_EM 0
#define ANDOR_VALUE_AMPLIFIER_CON 1
// Preamp gain
#define ANDOR_VALUE_PREAMP_GAIN_1_0 0
#define ANDOR_VALUE_PREAMP_GAIN_2_4 1
#define ANDOR_VALUE_PREAMP_GAIN_5_1 2
// Trigger
#define ANDOR_VALUE_TRIGGER_INTERNAL  0
#define ANDOR_VALUE_TRIGGER_EXTERNAL 1
#define ANDOR_VALUE_TRIGGER_FAST_EXTERNAL -1 // Combination of external and SetFastExtTrigger
#define ANDOR_VALUE_TRIGGER_EXTERNAL_START 6
#define ANDOR_VALUE_TRIGGER_EXTERNAL_EXPOSURE  7
#define ANDOR_VALUE_TRIGGER_SOFTWARE  10

//
// Ximea defines
//
#define XIMEA_MOVIE_HEADER_LENGTH 240
#define XIMEA_TRIGGER_INTERNAL 0
#define XIMEA_TRIGGER_EXTERNAL 1
#define XIMEA_TRIGGER_SOFTWARE 3

//
// Common camera frame struct
//
struct camera_save_struct {
	//
	// Common stuff
	//
	uint32_t magic; // 'AndO'
	uint32_t version;
	uint32_t type; // Camera type
	uint32_t pixelmode; // Pixel mode
	uint32_t length_header; // Header data in bytes ( Everything except image data )
	uint32_t length_data; // Total data length in bytes;
};

//
// IIDC movie frame struct
//
union iidc_save_feature_value {
	uint32_t value;
	float absvalue;
};

struct iidc_save_struct {
	//
	// Common stuff
	//
	uint32_t magic; // 'TemI'
	uint32_t version;
	uint32_t type; // Camera type
	uint32_t pixelmode; // Pixel mode
	uint32_t length_header; // Header data in bytes ( Everything except image data )
	uint32_t length_data; // Total data length in bytes;

	//
	// Camera specific stuff
	//
	// Camera properties
	uint64_t i_guid;
	uint32_t i_vendor_id;
	uint32_t i_model_id;

	// Frame properties
	uint32_t i_video_mode;
	uint32_t i_color_coding;

	uint64_t i_timestamp; // microseconds

	uint32_t i_size_x_max; // Sensor size
	uint32_t i_size_y_max;
	uint32_t i_size_x; // Selected region
	uint32_t i_size_y;
	uint32_t i_pos_x;
	uint32_t i_pos_y;

	uint32_t i_pixnum; // Number of pixels
	uint32_t i_stride; // Number of bytes per image line
	uint32_t i_data_depth;  // Number of bits per pixel.

	uint32_t i_image_bytes; // Number of bytes used for the image (image data only, no padding)
	uint64_t i_total_bytes; // Total size of the frame buffer in bytes. May include packet multiple padding and intentional padding (vendor specific)

	// Features
	uint32_t i_brightness_mode; // Current mode
	union iidc_save_feature_value i_brightness; // Can be also float if mode is IIDC_FEATURE_MODE_ABSOLUTE (1<<2)

	uint32_t i_exposure_mode;
	union iidc_save_feature_value i_exposure;

	uint32_t i_gamma_mode;
	union iidc_save_feature_value i_gamma;

	uint32_t i_shutter_mode;
	union iidc_save_feature_value i_shutter;

	uint32_t i_gain_mode;
	union iidc_save_feature_value i_gain;

	uint32_t i_temperature_mode;
	union iidc_save_feature_value i_temperature;

	uint32_t i_trigger_delay_mode;
	union iidc_save_feature_value i_trigger_delay;

	int32_t i_trigger_mode;

	// Advanced features
	uint32_t i_avt_channel_balance_mode;
	int32_t i_avt_channel_balance;

	// Image data
	uint8_t *data;
} __attribute__((__packed__));

//
// Andor movie frame struct
//
struct andor_save_struct {
	//
	// Common stuff
	//
	uint32_t magic; // 'TemI'
	uint32_t version;
	uint32_t type; // Camera type
	uint32_t pixelmode; // Pixel mode
	uint32_t length_header; // Header data in bytes ( Everything except image data )
	uint32_t length_data; // Total data length in bytes;

	//
	// Camera specific stuff
	//
	// Timestamp
	uint64_t a_timestamp_sec;
	uint64_t a_timestamp_nsec;

	// Frame properties
	int32_t a_x_size_max; // Sensor size
	int32_t a_y_size_max;
	int32_t a_x_start; // Selected size and positions
	int32_t a_x_end;
	int32_t a_y_start;
	int32_t a_y_end;
	int32_t a_x_bin;
	int32_t a_y_bin;

	// Camera settings
	int32_t a_ad_channel; // ADC
	int32_t a_amplifier; // EM or classical preamplifier
	int32_t a_preamp_gain; // Preamplifier gain
	int32_t a_em_gain; // EM gain
	int32_t a_hs_speed; // HS speed
	int32_t a_vs_speed; // VS speed
	int32_t a_vs_amplitude; // VS amplitude
	float a_exposure; // Exposure time in seconds
	int32_t a_shutter; // Shutter
	int32_t a_trigger; // Trigger
	int32_t a_temperature; // Temperature
	int32_t a_cooler; // Cooler
	int32_t a_cooler_mode; // Cooler mode
	int32_t a_fan; // Fan

	//
	// Image data
	//
	uint8_t *data;
} __attribute__((__packed__));

//
// Ximea movie frame struct
//
struct ximea_save_struct {
	//
	// Common stuff
	//
	uint32_t magic; // 'TemI'
	uint32_t version;
	uint32_t type; // Camera type
	uint32_t pixelmode; // Pixel mode
	uint32_t length_header; // Header data in bytes ( Everything except image data )
	uint32_t length_data; // Total data length in bytes;

	//
	// Camera specific stuff
	//
	char x_name[100]; // Camera name
	uint32_t x_serial_number; // Serial number

	// Timestamp
	uint64_t x_timestamp_sec;
	uint64_t x_timestamp_nsec;

	// Sensor
	uint32_t x_size_x_max; // Sensor size
	uint32_t x_size_y_max;
	uint32_t x_size_x; // Selected region
	uint32_t x_size_y;
	uint32_t x_pos_x;
	uint32_t x_pos_y;

	//
	// Features
	//
	uint32_t x_exposure; // Exposure [us]
	float x_gain; // Gain [dB]
	uint32_t x_downsampling; // Downsampling, 1 1x1, 2 2x2
	uint32_t x_downsampling_type; // 0 binning, 1 skipping
	uint32_t x_bpc; // Bad Pixels Correction, 0 disabled, 1 enabled
	uint32_t x_lut; // Look up table, 0 disabled, 1 enabled
	uint32_t x_trigger; // Trigger

	// Automatic exposure/gain
	uint32_t x_aeag; // 0 disabled, 1 enabled
	float x_aeag_exposure_priority; // Priority of exposure versus gain 0.0 1.0
	uint32_t x_aeag_exposure_max_limit; // Maximum exposure time [us]
	float x_aeag_gain_max_limit; // Maximum gain [dB]
	uint32_t x_aeag_average_intensity; // Average intensity level [%]

	// High dynamic range
	uint32_t x_hdr; // 0 disabled, 1 enabled
	uint32_t x_hdr_t1; // Exposure time of the first slope [us]
	uint32_t x_hdr_t2; // Exposure time of the second slope [us]
	uint32_t x_hdr_t3; // Exposure time of the third slope [us]
	uint32_t x_hdr_kneepoint1; // Kneepoint 1 [%]
	uint32_t x_hdr_kneepoint2; // Kneepoint 2 [%]

	//
	// Image data
	//
	uint8_t *data;
} __attribute__((__packed__));

struct frame_info{
    int size_x;
    int size_y;
    uint8_t depth;
};

#define NAMELENGTH 100

//read the .movie file, store frame in buffer
//buffer size_x*size_y*2*sizeof(uint8_t)
int load_frame(uint8_t * imagebuf, struct frame_info * frinfo, char * filename, int framenum){
  	int j, index;
	FILE *moviefile;
	long offset;
	bool found;
	uint32_t magic;
	struct camera_save_struct camera_frame;
	struct iidc_save_struct iidc_frame;
	struct andor_save_struct andor_frame;
	struct ximea_save_struct ximea_frame;
	int bpp;
	uint32_t size_x, size_y;
	char c;

    //Open movie file
    if ( !( moviefile = fopen( filename, "rb" ) ) ) {
    	printf( "Couldn't open movie file.\n" );
    	return -5;
	}

	// Find the beginning of binary data, it won't work if "TemI" is written in the header.
	offset = 0;
	found = false;
	while ( fread( &magic, sizeof( uint32_t ), 1, moviefile ) == 1 ) {
		if ( magic == CAMERA_MOVIE_MAGIC ) {
			found = true;
			break;
		}
		offset++;
		fseek( moviefile, offset, SEEK_SET );
	}
	// Go to the beginning of frames
	fseek( moviefile, offset, SEEK_SET );

	if ( found ) {
		index = 0;

		// Go through all the movie
		while ( fread( &camera_frame, sizeof( struct camera_save_struct ), 1, moviefile ) == 1 ) { // Check for the end of the file
			if ( camera_frame.magic != CAMERA_MOVIE_MAGIC ) {
				offset = ftell( moviefile );
				printf( "Wrong magic at offset %lu\n", offset );
				exit( EXIT_FAILURE );
			}
			if ( camera_frame.version != CAMERA_MOVIE_VERSION ) {
				printf( "Unsuported version %u\n", camera_frame.version );
				exit( EXIT_FAILURE );
			}
			// Go to the beginning of the frame for easier reading
			fseek( moviefile, -sizeof( struct camera_save_struct ), SEEK_CUR );

			// Read the header
			if ( camera_frame.type == CAMERA_TYPE_IIDC ) {
				if ( fread( &iidc_frame, IIDC_MOVIE_HEADER_LENGTH, 1, moviefile ) != 1 ) {
					offset = ftell( moviefile );
					printf( "Corrupted header at offset %lu\n", offset );
					exit( EXIT_FAILURE );
				}
				size_x = iidc_frame.i_size_x;
				size_y = iidc_frame.i_size_y;
				//printf( "%lu\n", iidc_frame.i_timestamp );
			}
			else if ( camera_frame.type == CAMERA_TYPE_ANDOR ) {
				if ( fread( &andor_frame, ANDOR_MOVIE_HEADER_LENGTH, 1, moviefile ) != 1 ) {
					offset = ftell( moviefile );
					printf( "Corrupted header at offset %lu\n", offset );
					exit( EXIT_FAILURE );
				}
				size_x = ( andor_frame.a_x_end - andor_frame.a_x_start + 1 ) / andor_frame.a_x_bin;
				size_y = ( andor_frame.a_y_end - andor_frame.a_y_start + 1 ) / andor_frame.a_y_bin;
			}
			else if ( camera_frame.type == CAMERA_TYPE_XIMEA ) {
				if ( fread( &ximea_frame, XIMEA_MOVIE_HEADER_LENGTH, 1, moviefile ) != 1 ) {
					offset = ftell( moviefile );
					printf( "Corrupted header at offset %lu\n", offset );
					exit( EXIT_FAILURE );
				}
				size_x = ximea_frame.x_size_x;
				size_y = ximea_frame.x_size_y;
			}
			else {
				printf( "Unsuported camera.\n" );
				return -1;
			}

			// Read the data
			if ( fread( imagebuf, camera_frame.length_data, 1, moviefile ) != 1 ) {
				offset = ftell( moviefile );
				printf( "Corrupted data at offset %lu\n", offset );
				exit( EXIT_FAILURE );
			}
            //TODO: this is very inefficient for videos (ideally read the 3 files in sync and merge)
            if(index == framenum){
                // Convert to little endian data
        		if ( camera_frame.pixelmode == CAMERA_PIXELMODE_MONO_16BE ) {
		        	for ( j = 0; j < size_x * size_y; j++ ) {
		    		    c = imagebuf[2*j];
		    		    imagebuf[2*j] = imagebuf[2*j+1];
		    		    imagebuf[2*j+1] = c;
		    	    }
		        }
                // Data depth
        		if ( camera_frame.pixelmode == CAMERA_PIXELMODE_MONO_8 )
            		bpp = 1;
	        	else
	        		bpp = 2;
                //we also need extra data to produces tiffs
                frinfo->size_x = size_x;
                frinfo->size_y = size_y;
                frinfo->depth = bpp;
                return 0;
            }
            index++;
        }
    }
    printf("!found\n");
     return -1;
}

//create green, blue, out paths from red
int rgb_names(char * red, char * f_g, char * f_b, char * f_out){
    char * r_pos = strstr(red,"_R");
    //get lengths of first and second parts
    int fp_len = r_pos-red;
    int sp_len = strlen(red) - fp_len - 2; //strlen(_red) = 4
    memcpy(f_g, red, fp_len*sizeof(char) );
    memcpy(f_g+fp_len*sizeof(char), "_G", 3*sizeof(char));
    memcpy(f_g+(fp_len+2)*sizeof(char), red+fp_len+2, (sp_len+1)*sizeof(char));
//    printf("%s\n", f_g);
    memcpy(f_b, red, fp_len*sizeof(char));
    memcpy(f_b+fp_len*sizeof(char), "_B", 3*sizeof(char));
    memcpy(f_b+(fp_len+2)*sizeof(char), red+fp_len+2, (sp_len+1)*sizeof(char));
//    printf("%s\n", f_b);
    memcpy(f_out, red, fp_len*sizeof(char));
    memcpy(f_out+fp_len*sizeof(char), red+fp_len+2, sizeof(char)*(sp_len-5)); //strip movie
    memcpy(f_out+(fp_len+sp_len-5)*sizeof(char),"tiff",5*sizeof(char));//include \0;
//    printf("%s\n", f_out);
    return 0;
}

//save the RGB tiff, rescale the values by color_rescale
//all images must have the same dimensions and depth!
int save_tiff(char * filename, uint8_t * buffer_r, uint8_t * buffer_g, uint8_t  * buffer_b,
             struct frame_info * frinfo, float * color_rescale){

		// Save the image in TIFF format
		// Create a new image file
       	TIFF *image;
    	if( ( image = TIFFOpen( filename, "w" ) ) == NULL ) {
			printf( "Could not open %s for writing\n", filename );
				exit( EXIT_FAILURE );
		}
    	// We need to set some values for basic tags before we can add any data
		TIFFSetField( image, TIFFTAG_IMAGEWIDTH, frinfo->size_x );
		TIFFSetField( image, TIFFTAG_IMAGELENGTH, frinfo->size_y );
		TIFFSetField( image, TIFFTAG_BITSPERSAMPLE, 8 * frinfo->depth );
		TIFFSetField( image, TIFFTAG_SAMPLESPERPIXEL, 3 );
		TIFFSetField( image, TIFFTAG_ROWSPERSTRIP, frinfo->size_y );

		TIFFSetField( image, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
		TIFFSetField( image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
		TIFFSetField( image, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
		TIFFSetField( image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG );

        //prepare the RGB image buffer
        uint8_t * imagebuf = (uint8_t *) malloc(3*frinfo->size_y*frinfo->size_x*frinfo->depth);
        uint8_t * vals = (uint8_t *) malloc(3*frinfo->depth);
        for(int i=0; i<frinfo->size_x*frinfo->size_y;i++){//for every pixel
            if(frinfo->depth == 2){
                ((uint16_t *) vals)[0] = (uint16_t) ((uint16_t *) buffer_r)[i]*color_rescale[0];
                ((uint16_t *) vals)[1] = (uint16_t) ((uint16_t *) buffer_g)[i]*color_rescale[1];
                ((uint16_t *) vals)[2] = (uint16_t) ((uint16_t *) buffer_b)[i]*color_rescale[2];
                //printf("%i,%i,%i,%i,%i,%i\n", vals[0],vals[1],vals[2],vals[3],vals[4],vals[5]);
            }else if(frinfo->depth==1){
                vals[0] = (uint8_t) buffer_r[frinfo->depth*i]*color_rescale[0];
                vals[1] = (uint8_t) buffer_g[frinfo->depth*i]*color_rescale[1];
                vals[2] = (uint8_t) buffer_b[frinfo->depth*i]*color_rescale[2];
            }else{
                printf("Unsupported depth\n");
                return -1;
            }
            memcpy(&imagebuf[3*i*frinfo->depth], vals, 3*frinfo->depth);
        }

		// Write the information to the file
		TIFFWriteEncodedStrip(image, 0, imagebuf, 3*frinfo->size_x * frinfo->size_y * frinfo->depth );
		// Close the file
		TIFFClose( image );
        free(imagebuf);
        free(vals);
        return 0;
}

int globerr(const char *path, int eerrno)
{
	printf(stderr, "%s: %s\n", path, strerror(eerrno));
	return 0;	/* let glob() keep going */
}

double average(uint8_t * data, struct frame_info * fi){
    double r=0;
    for(int i=0; i<fi->size_x*fi->size_y; i++){
        if(fi->depth ==2) r+= data[2*i]+data[2*i+1]*256; //big endian?
        else r+=data[i];
    }
    return r/(fi->size_x*fi->size_y);
}

int main( int argc, char *argv[] )
{
	if ( argc  < 3 ) {
		printf( "Wrong command.\n\"rgb_merge r.movie g.movie b.movie [frame]\"\n" );
        printf( "Wrong command.\n\"rgb_merge --all folder (calibration_red.movie)\n" );
		exit( EXIT_FAILURE );
	}


    //allocate buffers
    uint8_t * red = (uint8_t *)malloc( 2000*2000*2 );
    struct frame_info * red_f = (struct frame_info *)malloc(sizeof(struct frame_info));
    uint8_t * green = (uint8_t *)malloc( 2000*2000*2 );
    struct frame_info * green_f = (struct frame_info *)malloc(sizeof(struct frame_info));
    uint8_t * blue = (uint8_t *)malloc( 2000*2000*2 );
    struct frame_info * blue_f = (struct frame_info *)malloc(sizeof(struct frame_info));
	if ( red == NULL || green == NULL || blue == NULL ) {
		printf( "Couldn't allocate enough memory.\n" );
		exit( EXIT_FAILURE );
	}

    if (strcmp(argv[1],"--all")==0){
        char f_g[200];
        char f_b[200];
        char f_out[200];
        float colors[3]={1.0f,1.0f,1.0f};
        if(argc ==  4){
            rgb_names(argv[3], f_g, f_b, f_out);
            if(load_frame(red,red_f, argv[3], 1)||
               load_frame(green,green_f, f_g, 1)||
               load_frame(blue,blue_f, f_b, 1)){
                printf("Failed to load calibration info, skipping\n");
            }else{
                //average the intensities
                double ra=average(red, red_f);
                double ga=average(green, green_f);
                double ba=average(blue, blue_f);
                //double target=ra > ga ? (ra > ba ? ra : ba) : (ba > ga ? ba : ga);
                double target = 45000;
                colors[0] = (float) target/ra;
                colors[1] = (float) target/ga;
                colors[2] = (float) target/ba; //(all>=1, might clip but shouldn't)
                printf("Colour calibration RGB: %.3f %.3f %.3f", ra, ga, ba);//colors[1], colors[2]);
            }
        }
        glob_t results;
        char pat[200];
        pat[0]='\0';
        strcat(pat, argv[2]);
        strcat(pat, "/*_R*.movie");
        printf("%s\n", pat);
        int ret;
        if ((ret=glob(pat, 0, globerr, &results))){
            printf("Failed to find movies: %i \n", ret);
            globfree(& results);
            return -1;
        }
        for (int i = 0; i < results.gl_pathc; i++){
            printf("Processing: %s\n", results.gl_pathv[i]);
            rgb_names(results.gl_pathv[i], f_g, f_b, f_out);

            if(load_frame(red,red_f, results.gl_pathv[i], 0)||
               load_frame(green,green_f, f_g, 0)||
               load_frame(blue,blue_f, f_b, 0)){
                printf("Failed to load, skipping\n");
                continue;
            }
            save_tiff(f_out, red, green, blue, red_f, colors);
            printf("saved.\n");
        }

        globfree(& results);
        return 0;
    }

    int framenum;
    if(argc == 5)
        framenum = atoi(argv[4]);
    else
        framenum = 0;

    load_frame(red,red_f, argv[1], framenum);
    load_frame(green,green_f, argv[2], framenum);
    load_frame(blue,blue_f, argv[3], framenum);
    float colors[3]={1.0f,0.8f,0.4f};
    save_tiff("test.tiff", red, green, blue, red_f, colors);
	return 0;
}

