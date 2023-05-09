## Automated Ethogram GEnerAtioN (aegean) framework

Code to automatically generate ethograms and model behavior. This is traditionally the first attempt to what the **"Fish INteraction moDeling ([find](https://github.com/epfl-mobots/find))"** framework is doing. However, **aegean** is aimed to be faster and more efficient and hopefully it will some day replace **find** and/or **find** will use python bindings from **aegean** for some of its functionalities. 

**Notice:** This repo will undergo significant restructuring. While I will try to not break much, this is fairly unlikely...

### Dependencies 

1. boost (developing at 1.82)
2. eigen3
3. **[optional]** oneapiTBB

### Compilation 

**1.** First configure the code

```console
$ ./waf configure 
```

**2.** Then compile the examples
```console
$ ./waf 
```

**[optional] 3.** If you have external modules you will need to adapt the procedure as follows:

**1.** Configure module

```console
$ ./waf configure -j --exp EXT_MODULE_FOLDER_NAME
```

**2.** Compile module

```console
$ ./waf --exp EXT_MODULE_FOLDER_NAME
```

### Examples

You can find multiple examples under ``src/examples``. See some below:

**1.** Kmeans clustering

```console
$ ./build/examples/kmeans_example
```

**2.** Density-peak clustering

```console
$ ./build/examples/clusterdp_example
```

**3.** Histrogram example

```console
$ ./build/examples/histogram_example
```

## ToDos:
- [ ] Restructure repo
  - [ ] Refactor features and correct mistakes
  - [ ] Integrate particle_simu as submodule (?)
- [ ] Remove boost as a dependency (this is a biggy and might not even want it)
- [ ] Push for C++20 throughout the code
  - [ ] There are various C++20 improvements to be made and perhaps it's time to push for using the standard 
