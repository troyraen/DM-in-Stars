# work_standard_makefile

# change this as necessary.  see utils/makefile_header for definitions.

include $(MESA_DIR)/utils/makefile_header

LOAD_MESA = $(LOAD_EXTRAS1) $(LOAD_MESA_STAR) $(LOAD_EXTRAS)

#################################################################

ifndef STAR
STAR = star
endif

# STAR_WORK_OBJS defined in makefile_header
OBJS = $(STAR_WORK_OBJS) run.o DM_module.o DM_num.o nrtype.o nrutil.o

WORK_DIR = ..
WORK_SRC_DIR = $(WORK_DIR)/src
STAR_JOB_DIR = $(MESA_DIR)/star/job

$(STAR) : $(OBJS)
	$(LOADER) $(FCopenmp) -o $(WORK_DIR)/$(STAR) $(OBJS) $(LOAD_MESA)

#################################################################

# WORK_COMPILE defined in makefile_header

DM_module.o: $(WORK_SRC_DIR)/DM/DM_module.f90 DM_num.o
	$(WORK_COMPILE) $<

DM_num.o: $(WORK_SRC_DIR)/DM/DM_num.f90 nrtype.o nrutil.o
	$(WORK_COMPILE) $<

nrtype.o: $(WORK_SRC_DIR)/DM/nrtype.f90
	$(WORK_COMPILE) $<

nrutil.o: $(WORK_SRC_DIR)/DM/nrutil.f90
	$(WORK_COMPILE) $<


ifeq ($(SKIP_NAN_TRAPS),YES)
run.o: $(WORK_SRC_DIR)/run.f
	$(WORK_COMPILE) $<
else
run.o: $(WORK_SRC_DIR)/run.f
	$(WORK_COMPILE) $(FCtrapNANs) $<
endif

run_star_extras.o: $(WORK_SRC_DIR)/run_star_extras.f DM_module.o DM_num.o
	$(WORK_COMPILE) $<

%.o: $(STAR_JOB_DIR)/%.f
	$(WORK_COMPILE) $<

clean:
	-@rm -f *.o *.mod $(WORK_DIR)/$(STAR)

remk:
	-@rm -f run.o $(WORK_DIR)/$(STAR)
