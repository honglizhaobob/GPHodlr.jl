var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = GPHodlr","category":"page"},{"location":"#GPHodlr","page":"Home","title":"GPHodlr","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for GPHodlr.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [GPHodlr]","category":"page"},{"location":"#GPHodlr.SSTPartition","page":"Home","title":"GPHodlr.SSTPartition","text":"(03/15/2023) \nHelper functions to preprocess Sea Surface Temperature (SST) data.\n\nThe raw data is stored in `.jld` files and are assumed to be \ndictionaries containing the following fields:\n\n\"lat\"=> an approximately uniform grid containing latitude locations,\nwhich is interpreted as the y-axis.\n\n\"lon\"=> longitudal recordings, interpreted as the x-axis.\n\n\"cloud_mask\"=> 2d matrix of size (ny x nx), containing 0's and \n1's, where 1's indicate the SST reading is masked by cloud. \n\n\"ssta\"=> 2d matrix of size (ny x nx), containing zero-mean SST\nanomaly readings. The `NaN` values represent land locations.\n\n\n\n\n\n","category":"type"}]
}
