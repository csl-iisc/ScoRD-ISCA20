/********************************************************************************************
 * Copyright (c) 2020 Indian Institute of Science
 * All rights reserved.
 *
 * Developed by:    Aditya K Kamath, Alvin George A.
 *                  Computer Systems Lab
 *                  Indian Institute of Science
 *                  https://csl.csa.iisc.ac.in/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of Computer Systems Lab, Indian Institute of Science, 
 *        nor the names of its contributors may be used to endorse or promote products 
 *        derived from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ********************************************************************************************/
/* scord.cc */

#include "scord.h"
#include "gpu-sim.h"
#include "mem_fetch.h"
#include "gpu-cache.h"
#include "l2cache.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/ptx_sim.h"
#include "../cuda-sim/cuda-sim.h"
#include "../option_parser.h"

#define ID_INVALID    ((unsigned)-1)

static std::map<int, scord_metadata_t> memorydata;
static std::map<int, scord_execdata_t> execdata;

scord_detection_logic gscord_detection;
scord_request_generator gscord_request;

#define EXPORT  /* to mark exported symbols */

// Config options
bool SCORD_ENABLED;
bool SCORD_PERF;
int GRANULARITY;
bool SCORD_RACE_EXIT;
bool SCORD_PRINT;
bool SCORD_SILENT;
bool MD_CACHING;
bool SCORD_STRONG_OPS;

void scord_reg_options(option_parser_t opp)
{
    option_parser_register(opp, "-scord_enabled", OPT_BOOL, &SCORD_ENABLED, 
               "whether race detection is switched on",
               "1");
    option_parser_register(opp, "-scord_perf", OPT_BOOL, &SCORD_PERF, 
            "whether performance version of race detection is switched on",
            "1");
    option_parser_register(opp, "-scord_granularity", OPT_INT32, &GRANULARITY, 
            "granularity of tracking for race detector for normal, bytes per metadata entry for md caching (def=4)",
            "4");
    option_parser_register(opp, "-scord_race_exit", OPT_BOOL, &SCORD_RACE_EXIT, 
            "whether to exit immediately on detection of a race",
            "0");
    option_parser_register(opp, "-scord_print", OPT_BOOL, &SCORD_PRINT, 
            "whether to print all debug outputs",
            "1");
    option_parser_register(opp, "-scord_silent", OPT_BOOL, &SCORD_SILENT, 
            "whether to print any debug outputs",
            "0");
    option_parser_register(opp, "-scord_metadata_caching", OPT_BOOL, &MD_CACHING, 
            "whether to limit the shadow to a fixed size",
            "0");
    option_parser_register(opp, "-scord_strong_ops", OPT_BOOL, &SCORD_STRONG_OPS, 
            "whether to require strong operations with fences",
            "1");
}

scord_md_s::scord_md_s()
{
    warpid = ID_INVALID;
    ctaid = ID_INVALID;
    blkfenceid = 0;
    devfenceid = 0;
    modified = true;
    shared = true;
    devshared = true;
    lockset = lockset_t();
    strong = false;
    syncid = 0;
    tag = 0;
}

scord_execdata_s::scord_execdata_s()
{
    warpid = ID_INVALID;
    ctaid = ID_INVALID;
    blkfenceid = 0;
    devfenceid = 0;
    lockset = lockset_t();
    syncid = 0;
    vol = false;
}

void scord_reinit_memorymetadata()
{
    memorydata.clear();
}

void scord_reinit_execmetadata()
{
    execdata.clear();
}

static bool initdone = false;
void scord_initialize()
{
    if(SCORD_ENABLED)
    {
        if (initdone)
            return;
        scord_reinit_memorymetadata();
        scord_reinit_execmetadata();
        initdone = true;
    }
}
#define CHECK_INITIALIZE()  scord_initialize()

/*************************************************************************************/
static inline bool is_data_addr(unsigned addr)
{
    return addr >= GLOBAL_HEAP_START && addr < 0xf0000000;
}
static inline unsigned CHECK_ADDR(addr_t addr) {
    addr -= GLOBAL_HEAP_START;
    if(!MD_CACHING)
        addr /= GRANULARITY;
    else
        addr /= 4;
    return addr;
}

EXPORT
void scord_kernel_restart(gpgpu_t *gpu)
{
    if(!SCORD_ENABLED)
        SCORD_PERF = 0;
    
    if(SCORD_PERF)
        gscord_request.reset(gpu);
    else
        initdone = false;
}

EXPORT
unsigned scord_ctaid(ptx_thread_info *thread)
{
    dim3 cta = thread->get_ctaid();
    dim3 gdim = thread->get_kernel().get_grid_dim();
    int clinear = cta.x + cta.y * gdim.x + cta.z * gdim.x * gdim.y;
    return clinear;
}

EXPORT
unsigned scord_warpid(ptx_thread_info *thread)
{
    dim3 thd = thread->get_tid();
    dim3 cdim = thread->get_kernel().get_cta_dim();
    int wlinear = thd.x + thd.y * cdim.x + thd.z * cdim.x * cdim.y;
    // Assumption: 32 threads per warp
    return wlinear >> 5;
}


unsigned get_execid(int cta, int warp)
{
    return warp + (cta << 6);
}

#define WARPID(t)     scord_warpid(t)
#define CTAID(t)      scord_ctaid(t)

#define VAL_TRACK_BY_TID  1
#define VAL_TRACK_BY_WID  2

#if CFG_TRACKING_UNIT == VAL_TRACK_BY_TID

//#define EXEC_INVALID   MAX_EXECS
#define EXEC_INVALID     ((unsigned)-1)
#define execid_t            threadid_t
#define EXECID(pti)         (THREADID(pti) + (CTAID(pti) << 12))
#define EXECID_MEMDATA(md)  ((md)->threadid)
#elif CFG_TRACKING_UNIT == VAL_TRACK_BY_WID
#define EXECID(pti)      (WARPID(pti) + (CTAID(pti) << 6))
#define EXEC_INVALID     ((unsigned)-1)
#define execid_t            warpid_t
unsigned scord_execid(ptx_thread_info *thread)
{
    return EXECID(thread);
}
#else
#error  not impl
#endif

/*************************************************************************************/
EXPORT
void scord_gmem_read(unsigned addr, unsigned size, ptx_thread_info *pti) // XXX
{
    if(SCORD_ENABLED)
    {
        warpid_t wid = WARPID(pti);
        ctaid_t  cid = CTAID(pti);
        execid_t xid = EXECID(pti);

        if (!is_data_addr(addr)) return;
        addr = CHECK_ADDR(addr);

        CHECK_INITIALIZE();

        const warp_inst_t *pI = (warp_inst_t *)pti->get_inst();

        scord_metadata_t *md = &memorydata[addr];
        scord_execdata_t *wd = &execdata[xid];
        wd->ctaid = cid;
        wd->warpid = wid;
        wd->atom       = pI->isatomic();
        wd->scope = (pI->get_atomic_scope() == 408 ? 1 : 0);
        wd->inst       = pI->pc;

        unsigned oxd = get_execid(md->ctaid, md->warpid);
        if(md->warpid == ID_INVALID)
            oxd = 0;
        gscord_detection.check_read_race(md, wd, execdata[oxd].blkfenceid, execdata[oxd].devfenceid);
    }
}

EXPORT
void scord_gmem_write(unsigned addr, unsigned size, ptx_thread_info *pti)
{
    if(SCORD_ENABLED)
    {
        warpid_t wid = WARPID(pti);
        ctaid_t  cid = CTAID(pti);
        execid_t xid = EXECID(pti);

        if (!is_data_addr(addr)) return;
        addr = CHECK_ADDR(addr);

        CHECK_INITIALIZE();

        const warp_inst_t *pI = (warp_inst_t *)pti->get_inst();

        scord_metadata_t *md = &memorydata[addr];
        scord_execdata_t *xd = &execdata[xid];
        xd->ctaid      = cid;
        xd->warpid     = wid;
        xd->atom       = pI->isatomic();
        xd->scope = (pI->get_atomic_scope() == 408 ? 1 : 0);
        xd->inst       = pI->pc;

        unsigned oxd = get_execid(md->ctaid, md->warpid);
        if(md->warpid == ID_INVALID)
            oxd = 0;
        gscord_detection.check_write_race(md, xd, execdata[oxd].blkfenceid, execdata[oxd].devfenceid);
        
    }
}

EXPORT
void scord_thread_blkexecbarrier(ptx_thread_info *pti)
{
    if(SCORD_ENABLED)
    {
        execid_t xid = EXECID(pti);

        CHECK_INITIALIZE();
        scord_execdata_t *xd = &execdata[xid];
        if(pti->get_hw_tid() % 32 == 0)
        {
            xd->syncid++;
            printf_scord("@@ SCORD syncthreads: ctaid=%d, wid=%d, syncid=%d\n", CTAID(pti), WARPID(pti), xd->syncid);
        }
    }
}

/* not called externally */
void scord_thread_blkfence(execid_t xid)
{
    CHECK_INITIALIZE();
    scord_execdata_t *xd = &execdata[xid];
    xd->blkfenceid++;
    if(SCORD_PERF)
        xd->blkfenceid &= 63;
}

/* not called externally */
void scord_thread_devfence(execid_t xid)
{
    CHECK_INITIALIZE();
    scord_execdata_t *xd = &execdata[xid];
    xd->devfenceid++;
    if(SCORD_PERF)
        xd->devfenceid &= 63;
}

EXPORT
void scord_thread_genericfence(ptx_thread_info *pti, unsigned level)
{
    if(SCORD_ENABLED)
    {
        execid_t xid = EXECID(pti);
        switch (level) {
            case 407: scord_thread_devfence(xid); break;
            case 408: scord_thread_blkfence(xid); break;
        }
        printf_scord("@@ SCORD fence: level=%d, ctaid=%d, wid=%d\n", level, CTAID(pti), WARPID(pti));
    }
}

void scord_thread_devatom(ptx_thread_info *pti, bool unlock, unsigned lockid)
{
    execid_t xid = EXECID(pti);
    printf_scord("@@ SCORD Device lock access: cta=%d, wid=%d, unlock=%d, lockid=%d\n", CTAID(pti), WARPID(pti), unlock, lockid);
    CHECK_INITIALIZE();
    scord_execdata_t *xd = &execdata[xid];
    if(unlock)
        xd->lockset.erase(lockid_t(lockid, 0));
    else
        xd->lockset.insert(lockid_t(lockid, 0));
}

void scord_thread_blkatom(ptx_thread_info *pti, bool unlock, unsigned lockid)
{
    execid_t xid = EXECID(pti);
    printf_scord("@@ SCORD Block lock access: cta=%d, wid=%d, unlock=%d, lockid=%d\n", CTAID(pti), WARPID(pti), unlock, lockid);
    CHECK_INITIALIZE();
    scord_execdata_t *xd = &execdata[xid];
    if(unlock)
        xd->lockset.erase(lockid_t(lockid, CTAID(pti) + 1));
    else
        xd->lockset.insert(lockid_t(lockid, CTAID(pti) + 1));
}

EXPORT
// Used for both implicit locking, with atomics, and explicit locking with CAS/Exch
void scord_thread_atomgeneric(ptx_thread_info *pti, unsigned level, bool unlock, unsigned lockid)
{
    if(SCORD_ENABLED)
    {
        switch(level) {
            case 0:   scord_thread_devatom(pti, unlock, lockid);
            case 408: scord_thread_blkatom(pti, unlock, lockid); break;
        }
    }
}


EXPORT
void scord_locks_enable(ptx_thread_info *pti, unsigned level)
{
    if(SCORD_ENABLED)
    {
        execid_t xid = EXECID(pti);
        printf_scord("@@ SCORD Device lock activate: cta=%d, wid=%d\n", CTAID(pti), WARPID(pti));
        CHECK_INITIALIZE();

        scord_execdata_t *xd = &execdata[xid];
        xd->lockset.enableLocks(level);
    }
}

/*************************************************************************************/
void scord_detection_logic::declare_race(char *msg, scord_metadata_t *md, scord_execdata_t *xd)
{
    if(!SCORD_SILENT)
    {
        printf("***************** SCORD RACE (PC=%x): %s ****************\n", xd->inst, msg);
        printf("\tData: wid:%d, bid=%d, m=%d, b_s=%d, d_s=%d, syncid=%d, devfence=%d, blkfence=%d\n", 
            md->warpid, md->ctaid, md->modified, md->shared, md->devshared, md->syncid, md->devfenceid, md->blkfenceid);
        printf("\tThread: wid:%d, bid=%d, syncid=%d, devfence=%d, blkfence=%d\n", 
            xd->warpid, xd->ctaid, xd->syncid, xd->devfenceid, xd->blkfenceid);
    }
    races[xd->inst]++;
    ++race_types[xd->inst][msg];
    ++races2;
    if(SCORD_RACE_EXIT)
        exit(SCORD_RACE_EXIT);
    else
    {
        if(!SCORD_SILENT)
            printf("SCORD Reset data\n");
        md->modified = true;
        md->shared = true;
        md->devshared = true;
    }
}

void scord_detection_logic::make_owner(scord_md_s *md, scord_execdata_s *xd, bool write)
{
    md->warpid      = xd->warpid;
    md->ctaid       = xd->ctaid;
    md->syncid      = xd->syncid;
    md->blkfenceid  = xd->blkfenceid;
    md->devfenceid  = xd->devfenceid;
    md->atom        = xd->atom;
    md->scope       = xd->scope;
    md->strong      = (xd->vol || xd->atom) && md->strong;
}

void scord_detection_logic::check_read_race(scord_metadata_t *md, scord_execdata_s *xd, fenceid_t blkfid, fenceid_t devfid)
{
    printf_scord("@@ SCORD read: wid=%d, bid=%d, m=%d, bs=%d, ds=%d, syncid=%d\n", md->warpid, md->ctaid, md->modified, md->shared, md->devshared, md->syncid);
    printf_scord("\tData: blkfenceid=%d, devfenceid=%d, nblkfenceid=%d, ndevfenceid=%d\n", md->blkfenceid, md->devfenceid, blkfid, devfid);
    printf_scord("\tReader: wid=%d, bid=%d, syncid=%d\n", xd->warpid, xd->ctaid, xd->syncid);

    update_stats(false, md->modified, md->shared, md->ctaid == xd->ctaid, !md->lockset.empty() || !xd->lockset.empty());

    if ((md->shared && md->modified && md->devshared) 
        || (md->ctaid == xd->ctaid && md->syncid != xd->syncid && !md->devshared)) 
    {
        printf_scord("\tReset access\n");
        make_owner(md, xd, 0);
        md->modified    = false;
        md->shared      = false;
        md->devshared   = false;
        md->lockset     = xd->lockset;
        md->strong      = (xd->atom || xd->vol);
        return;
    }

    
    if(md->warpid == xd->warpid && md->ctaid == xd->ctaid && !md->shared && !md->devshared)
    {
        if(md->modified && (md->blkfenceid != blkfid || md->devfenceid != devfid))
            md->modified = false;
        make_owner(md, xd, 0);
        md->strong      = (xd->atom || xd->vol);
        md->lockset     = xd->lockset;
        return;
    }

    // Previous atomic was block scope by a different block
    if(md->atom && md->scope && md->ctaid != xd->ctaid)
    {
        declare_race("Atomic with block scope, and differing blocks", md, xd);
        check_write_race(md, xd, blkfid, devfid);
        return;
    }

    if(md->modified && (md->warpid != xd->warpid || md->ctaid != xd->ctaid))
    {
        if (md->ctaid == xd->ctaid && md->blkfenceid == blkfid && md->devfenceid == devfid) {
            // owner thread should have executed a block/dev fence; else is a RACE
            declare_race("RAW: No tf for matching blocks", md, xd);
            check_read_race(md, xd, blkfid, devfid);
            return;
        } else if (md->ctaid != xd->ctaid && md->devfenceid == devfid) {
            // owner thread has not executed a devfence
            declare_race("RAW: No tf for different blocks", md, xd);
            check_read_race(md, xd, blkfid, devfid);
            return;
        }
        else if(SCORD_STRONG_OPS && (md->strong == false || !(xd->atom || xd->vol))) {
            // threads should use strong accesses for synchronization
            declare_race("RAW: No strong op for read/write", md, xd);
            check_read_race(md, xd, blkfid, devfid);
            return;
        }
    }

    // Unprotected access, use barrier race detection
    if(md->lockset.empty() && xd->lockset.empty())
    {
        printf_scord("\tUnprotected access\n");
        if(md->modified)
        {
            if(md->warpid != xd->warpid || md->ctaid != xd->ctaid)
                md->modified = false;
            md->shared = false;
            md->devshared = false;
            make_owner(md, xd, 0);
            return;
        }
        else if(md->ctaid == xd->ctaid && !md->devshared)
        {
            md->shared = true; 
            md->devshared = false;   
        }   
        else
        {
            md->devshared = true;
            md->shared = false;
        }
    }
    // Protected access, use lockset race detection
    else
    {
        printf_scord("\tProtected access, data locked=%d, reader locked=%d, common=%d\n", 
            !md->lockset.empty(), !xd->lockset.empty(), !intersect_locks(md->lockset, xd->lockset).empty());
        
        // Data has been written, no locks are protecting
        if(md->modified && intersect_locks(md->lockset, xd->lockset).empty())
        {
            declare_race("RAW: Missing lock", md, xd);
            check_read_race(md, xd, blkfid, devfid);
            return;
        }

        // Store locks protecting the data
        if(md->lockset.empty())
            md->lockset = xd->lockset;
        else if(!xd->lockset.empty())
            md->lockset = intersect_locks(md->lockset, xd->lockset);
        if(md->ctaid != xd->ctaid)
        {
            md->devshared == true;
            md->shared = false;
        }
        else if(md->warpid != xd->warpid && !md->devshared)
        {
            md->devshared == false;
            md->shared = true;
        }
    }
}
void scord_detection_logic::check_write_race(scord_metadata_t *md, scord_execdata_s *xd, fenceid_t blkfid, fenceid_t devfid)
{
    printf_scord("@@ SCORD write: wid=%d, bid=%d, m=%d, bs=%d, ds=%d, syncid=%d\n", md->warpid, md->ctaid, md->modified, md->shared, md->devshared, md->syncid);
    printf_scord("\tData: blkfenceid=%d, devfenceid=%d, nblkfenceid=%d, ndevfenceid=%d\n", md->blkfenceid, md->devfenceid, blkfid, devfid);
    printf_scord("\tWriter: wid=%d, bid=%d, syncid=%d\n", xd->warpid, xd->ctaid, xd->syncid);

    update_stats(true, md->modified, md->shared, md->ctaid == xd->ctaid, !md->lockset.empty() || !xd->lockset.empty());

    if ((md->shared && md->modified && md->devshared) 
        || (md->ctaid == xd->ctaid && md->syncid != xd->syncid && !md->devshared)) {
        printf_scord("\tReset access\n");
        make_owner(md, xd, 1);
        md->modified    = true;
        md->shared      = false;
        md->devshared   = false;
        md->lockset     = xd->lockset;
        md->strong      = (xd->atom || xd->vol);
        return;
    }

    if(md->warpid == xd->warpid && md->ctaid == xd->ctaid && !md->shared && !md->devshared)
    {
        make_owner(md, xd, 1);
        md->modified = true;
        md->lockset     = xd->lockset;
        md->strong      = (xd->atom || xd->vol);
        return;
    }

    // Previous atomic was block scope by a different block
    if(md->atom && md->scope && md->ctaid != xd->ctaid)
    {
        declare_race("Atomic with block scope, and differing blocks", md, xd);
        check_write_race(md, xd, blkfid, devfid);
        return;
    }
    else if(md->atom && xd->atom)
    {
        printf_scord("\tAtomic, reset access\n");
        make_owner(md, xd, 1);
        md->modified = true;
        return;
    }

    if(md->warpid != xd->warpid || md->ctaid != xd->ctaid || md->devshared || md->shared)
    {
        if (md->ctaid == xd->ctaid && (md->blkfenceid == blkfid || md->devshared) && md->devfenceid == devfid) {
            // owner thread should have executed a block/dev fence; else is a RACE
            declare_race("WAR: No tf for matching blocks", md, xd);
            check_read_race(md, xd, blkfid, devfid);
            return;
        } else if (md->ctaid != xd->ctaid && md->devfenceid == devfid) {
            // owner thread has not executed a devfence
            declare_race("WAR: No tf for different blocks", md, xd);
            check_read_race(md, xd, blkfid, devfid);
            return;
        }
        else if(SCORD_STRONG_OPS && (md->strong == false || !(xd->atom || xd->vol))) {
            // threads should use strong accesses for synchronization
            declare_race("WAR: No strong op for read/write", md, xd);
            check_write_race(md, xd, blkfid, devfid);
            return;
        }
    }
    
    // Unprotected access, use barrier race detection
    if(md->lockset.empty() && xd->lockset.empty())
    {
        printf_scord("\tUnprotected access\n");
        make_owner(md, xd, 1);
        md->devshared = false;
        md->shared = false;
    }
    // Protected access, use lockset race detection
    else
    {
        printf_scord("\tProtected access, data locked=%d, reader locked=%d, common=%d\n", 
            !md->lockset.empty(), !xd->lockset.empty(), !intersect_locks(md->lockset, xd->lockset).empty());
        // Data has been accessed, no locks are protecting
        if(intersect_locks(md->lockset, xd->lockset).empty())
        {
            declare_race("WAR/W: Missing lock", md, xd);
            check_write_race(md, xd, blkfid, devfid);
            return;
        }
        
        make_owner(md, xd, 1);
        md->modified = true;
        
        if(md->ctaid != xd->ctaid)
        {
            md->devshared = true;
            md->shared = false;
        }
        else if(md->warpid != xd->warpid && !md->devshared)
        {
            md->devshared = false;
            md->shared = true;
        }

        // Store locks protecting the data
        md->lockset = intersect_locks(md->lockset, xd->lockset);
    }
    md->modified = true;
}

void scord_detection_logic::print_stats()
{
        int total = 0;
        printf("Detection stats:\n");
        for(int i = 0; i < STAT_PARAMS; ++i)
        {
            if(request_stats[i])
            {
                printf("%s, ", (i & 1 ? "Write" : "Read"));
                printf("m=%s, ", (i & 2 ? "true" : "false"));
                printf("s=%s, ", (i & 4 ? "true" : "false"));
                printf("bid_match=%s, ", (i & 8 ? "true" : "false"));
                printf("locked=%s: ", (i & 16 ? "true" : "false"));
                printf("%d\n", request_stats[i]);
                total += request_stats[i];
            }
        }
        printf("Total: %d\n\n", total);
        for(std::map<int, int>::iterator r = races.begin(); r != races.end(); ++r)
        {
            printf("SCORD Race races=%d insn=", r->second);
            ptx_print_insn(r->first, stdout);
            printf("\n");
            for(std::map<std::string, int>::iterator it = race_types[r->first].begin(); it != race_types[r->first].end(); ++it)
            {
                if(it->second != 0)
                    printf("SCORD Racetype (%s): %d\n", it->first.c_str(), it->second);
            }
            printf("\n");
        }
        printf("Total races: %d (%d)\n", races.size(), races2);
    }

/*************************************************************************************/

void scord_request_generator::reset(gpgpu_t *gpu)
{
    if(SCORD_PERF)
    {
        unsigned long long data_size = gpu->get_allocated_mem_size();
        printf("SCORD reset: Size of allocated global memory = %d\n", data_size);
        if(shadow_mem == NULL)
        {
            if(!MD_CACHING)
            {
                // Compute shadow data required for allocated global memory
                data_size /= GRANULARITY;
                data_size *= SCORD_SHADOW_SIZE;
                shadow_mem = (unsigned long)gpu->gpu_malloc(data_size);
                shadow_size = data_size;
            }
            else
            {
                data_size += (GRANULARITY - 1); // Round up
                data_size /= GRANULARITY;
                data_size *= SCORD_SHADOW_SIZE;
                shadow_mem = (unsigned long)gpu->gpu_malloc(data_size);
                shadow_size = data_size;
            }
        }
        printf("\tSCORD shadow data size: %d bytes\n", shadow_size);
        printf("\tShadow data starts at %x\n", shadow_mem);
        // Set all fields to 0
        char *c = new char[shadow_size];
        memset(c, 0, shadow_size);
        gpu->get_global_memory()->write(shadow_mem, shadow_size, c, NULL, NULL);
        delete c;
        // Set modified and shared fields to true
        for(int i = 0; i < shadow_size / SCORD_SHADOW_SIZE; ++i)
        {
            gpu->gpu_memset(shadow_mem + i * SCORD_SHADOW_SIZE + 4, 3, 1);
            gpu->gpu_memset(shadow_mem + i * SCORD_SHADOW_SIZE + 5, 128, 1);
        }
        scord_reinit_execmetadata();
    }
}

void scord_request_generator::generate_mem_request(mem_fetch **mf_ptr, memory_sub_partition **l2_cache, unsigned long long cycle)
{
    if(SCORD_PERF)
    {
        mem_fetch *mf = *mf_ptr;

        if(!mf)
            return;

        // Threadfence update
        if(mf->is_scord_data() && !mf->has_scord_metadata())
        {   
            printf_scord("SCORD generate_mem_request: Received request to increment fence for %x, scope = %d\n", 
                mf->get_addr(), *mf->get_data_array());
            // GLOBAL_OPTION = 407,
            // CTA_OPTION = 408,
            // Update threadfence
            switch(*mf->get_data_array())
            {
                case 407:
                    scord_thread_devfence(mf->get_addr());
                break;
                case 408:
                    scord_thread_blkfence(mf->get_addr());
                break;
                default:
                    printf("SCORD Error, invalid threadfence scope %d\n", *mf->get_data_array());
                    exit(127);
                break;
            }
            delete mf;
            *mf_ptr = NULL;
            return;
        }
        // Not a genuine request, ignore
        if(!mf->has_scord_metadata() || !is_data_addr(mf->get_addr()) || mf->get_addr() >= shadow_mem)
        {
            if(mf->is_scord_data())
            {
                delete mf;
                *mf_ptr = NULL;
            }
            return;
        }
        printf_scord("SCORD generate_mem_request: Received mf (%x) for addr=%x\n", mf, mf->get_addr());
        bool write = mf->is_write() || mf->isatomic();
        // Extract metadata from packet
        scord_execdata_t xd = extract_packet_metadata(mf);
        mf->has_scord_metadata() = false;
        // Coalesce requests for shadow memory
        std::map<unsigned, mem_fetch*> mem_reqs;
        for(auto i = mf->get_ldst_data()->begin(); i != mf->get_ldst_data()->end(); ++i)
        {
            if(!i->rd)
                continue;
            // Calculate address of shadow data
            addr_t addr = calculate_shadow_address(i->addr);
            if(MD_CACHING)
            {
                int tag = (i->addr - GLOBAL_HEAP_START) / 4;
                tag = tag / (shadow_size / SCORD_SHADOW_SIZE);
                xd.tag = tag;
            }
            mem_fetch *mf_req;
            // Already waiting on mem_fetch for same addr
            if(activeList.find(addr) != activeList.end())
            {
                // Store details into active list
                std::pair<bool, scord_execdata_s> ma = std::pair<bool, scord_execdata_s> (write, xd);
                activeList[addr].push(ma);
                continue;
            }
            // Memfetch already exists, append data
            if(mem_reqs.find(addr / 128) != mem_reqs.end())
            {
                mf_req = mem_reqs[addr / 128];
                // Do not append data for accesses within the same granularity
                bool flag = false;
                for(auto j = mf_req->get_ldst_data()->begin(); j != mf_req->get_ldst_data()->end(); ++j)
                {
                    if(addr == j->addr)
                    {
                        flag = true;
                        break;
                    }
                }
                if(flag)
                {
                    // Store details into active list
                    std::pair<bool, scord_execdata_s> ma = std::pair<bool, scord_execdata_s> (write, xd);
                    activeList[addr].push(ma);
                    continue;
                }
            }
            // Memfetch does not exist, create new one
            else
            {
                const mem_access_t ma = mem_access_t(GLOBAL_ACC_R, addr & ~(127), 128, false);
                mf_req = new mem_fetch(ma, &mf->get_inst(), mf->get_ctrl_size(), mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf->get_mem_config());
                mf_req->memspace = mf->memspace;
                mem_reqs[addr / 128] = mf_req;
            }
            // Add ldst data to mf
            ldst_data_alvin ldst = ldst_data_alvin();
            ldst.addr = addr;
            ldst.len = SCORD_SHADOW_SIZE;
            ldst.memspace = i->memspace;
            ldst.thread = i->thread;
            ldst.pI = i->pI;
            mf_req->get_ldst_data()->insert(ldst);
            printf_scord("\tCreated mf (%x) for shadow data of %x at %x\n", mf_req, i->addr, addr);

            // Store details into active list
            std::pair<bool, scord_execdata_s> ma = std::pair<bool, scord_execdata_s> (write, xd);
            activeList[addr].push(ma);
        }
        
        // Push generated mfs to L2 cache
        for(auto i = mem_reqs.begin(); i != mem_reqs.end(); ++i)
        {
            printf_scord("\tPushing mf (%x) for shadow data %x in subpartition %d\n", i->second, 
                i->second->get_addr(), i->second->get_sub_partition_id());
            l2_cache[i->second->get_sub_partition_id()]->push(i->second, cycle);
            ++req_in_flight;
            ++cur_reqs;
            max_reqs = (cur_reqs > max_reqs ? cur_reqs : max_reqs);
        }

        // Read hit
        if(mf->is_scord_data())
        {
            delete mf;
            *mf_ptr = NULL;
        }
    }
}

mem_fetch* scord_request_generator::process_mem_requests(memory_sub_partition **l2_cache, int line, unsigned long long cycle)
{
    if(SCORD_PERF)
    {
        mem_fetch *mf = l2_cache[line]->top();
        // Check if shadow data request
        while(mf && (mf->get_addr() >= shadow_mem && mf->get_addr() < (shadow_mem + shadow_size)))
        {
            l2_cache[line]->pop();
            // Writeback request for shadow mem, ignore
            if(mf->is_write())
            {
                --req_in_flight;
                delete mf;
                mf = l2_cache[line]->top();
                continue;
            }
            --cur_reqs;
            printf_scord("SCORD process_mem_req: Received mf (%x)\n", mf);
            const mem_access_t ma = mem_access_t(GLOBAL_ACC_W, mf->get_addr(), 0, true);
            mem_fetch *mf_req = new mem_fetch(ma, NULL, mf->get_ctrl_size(), mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf->get_mem_config());
            mf_req->memspace = mf->memspace;
            for(auto i = mf->get_ldst_data()->begin(); i != mf->get_ldst_data()->end(); ++i)
            {
                if(!i->rd)
                    continue;
                scord_metadata_t md = extract_shadow_metadata((unsigned char*)mf->get_data_array() + (i->addr % 128));
                while(!activeList[i->addr].empty())
                {
                    std::pair<bool, scord_execdata_s> ma = activeList[i->addr].front();
                    activeList[i->addr].pop();
                    unsigned xid = get_execid(md.ctaid, md.warpid);
                    int blkfenceid = execdata[xid].blkfenceid;
                    int devfenceid = execdata[xid].devfenceid;
                    if(MD_CACHING)
                    {
                        // Print debug statement
                        size_t offset = (i->addr - (size_t)shadow_mem) / SCORD_SHADOW_SIZE;
                        offset += shadow_size * ma.second.tag;
                        size_t addr = GLOBAL_HEAP_START + offset * GRANULARITY;
                        printf_scord("\tChecking race for data %x of %x, shadowatom=0x%x, execatom=0x%x,tag=%d\n", 
                            i->addr, addr, md.lockset.getVal(), ma.second.lockset.getVal(), ma.second.tag);
                        
                        if(ma.second.tag != md.tag)
                        {
                            printf_scord("SCORD Tag mismatch, resetting data\n");
                            md = scord_metadata_t();
                        }
                    }
                    else
                    {
                        size_t offset = (i->addr - (size_t)shadow_mem) / SCORD_SHADOW_SIZE;
                        size_t addr = GLOBAL_HEAP_START + offset * GRANULARITY;
                        printf_scord("\tChecking race for data %x of %x, shadowatom=0x%x, execatom=0x%x\n", 
                            i->addr, addr, md.lockset.getVal(), ma.second.lockset.getVal());
                    }

                    if(ma.first)
                    {
                        gscord_detection.check_write_race(&md, &ma.second, blkfenceid, devfenceid);
                    }
                    else
                    {
                        gscord_detection.check_read_race(&md, &ma.second, blkfenceid, devfenceid);
                    }
                    if(MD_CACHING)
                    {
                        md.tag = ma.second.tag;
                    }
                }
                activeList.erase(i->addr);
                scord_create_shadow_metadata(md, mf_req, i->addr);
                mf_req->get_ldst_data()->insert(*i);
            }
            printf_scord("Pushing mf (%x) to writeback modified shadow data\n", mf_req);
            l2_cache[mf_req->get_sub_partition_id()]->push(mf_req, cycle);
            delete mf;
            mf = l2_cache[line]->top();
        }
        return mf;
    }
    else
        return l2_cache[line]->top();
}

addr_t scord_request_generator::calculate_shadow_address(int addr)
{
    if(!MD_CACHING)
    {
        size_t offset = (addr - GLOBAL_HEAP_START) / GRANULARITY;
        return (size_t)shadow_mem + offset * SCORD_SHADOW_SIZE;
    }
    else
    {
        size_t offset = (addr - GLOBAL_HEAP_START) / 4;
        offset = offset % (shadow_size / SCORD_SHADOW_SIZE);
        return (size_t)shadow_mem + offset * SCORD_SHADOW_SIZE;
    }
}

scord_metadata_t scord_request_generator::extract_shadow_metadata(unsigned char *shadow)
{
    if(SCORD_PERF)
    {
        // (blkid (7) + syncid (8) + atomicid (16) + warpid (5) + modified (1) + {blk-}shared (1)
        // + dev-shared (1) + devfenceid (6) + blkfenceid (6) + atom (1) + scope (1) + tag (4)) / 8
        scord_metadata_t md;
        md.ctaid        = shadow[0] & 127;
        md.syncid       = shadow[1];
        unsigned long val = shadow[2];
        val           <<= BLOOM_BIN_SIZE;
        val            += shadow[3];
        md.lockset.setVal(val);
        md.warpid       = (shadow[4] >> 3) & 31;
        md.modified     = shadow[4] & 2;
        md.shared       = shadow[4] & 1;
        md.devshared    = shadow[5] & 128;
        md.strong       = shadow[5] & 64;
        md.devfenceid   = shadow[5] & 63;
        md.blkfenceid   = shadow[6] >> 2;
        md.atom         = shadow[6] & 2;
        md.scope        = shadow[6] & 1;
        md.tag          = (shadow[7] & 15);
        printf_scord("\tCTA=%d, Sync=%d, atomicid=0x%x, modified=%d, shared=%d\n", md.ctaid, 
            md.syncid, (shadow[4] << 8) + shadow[5], md.modified, md.shared);
        return md;
    }
}

void scord_request_generator::scord_create_shadow_metadata(scord_metadata_t md, mem_fetch *mf, int addr)
{
    if(SCORD_PERF)
    {
        // (blkid (7) + syncid (8) + atomicid (16) + warpid (5) + modified (1) + {blk-}shared (1)
        // + dev-shared (1) + devfenceid (6) + blkfenceid (6) + atom (1) + scope (1) + tag (4)) / 8
        unsigned char *shadow = (unsigned char*)mf->get_data_array() + (addr & 127);
        shadow[0]  = md.ctaid & 127;
        shadow[1]  = md.syncid & 255;
        unsigned long val = md.lockset.getVal();
        shadow[2]  = (val >> 8) & 255;
        shadow[3]  = val & 255;
        shadow[4]  = (md.warpid & 31) << 3;
        shadow[4] += (md.modified ? 2 : 0);
        shadow[4] += (md.shared ? 1 : 0);
        shadow[5]  = (md.devshared ? 128 : 0);
        shadow[5] += (md.strong ? 64 : 0);
        shadow[5] += md.devfenceid & 63;
        shadow[6]  = (md.blkfenceid & 63) << 2;
        shadow[6] += (md.atom ? 2 : 0);
        shadow[6] += (md.scope ? 1 : 0);
        shadow[7]  = (md.tag & 15);

        mf->set_data_valid(true);
        mf->set_data_size(SCORD_SHADOW_SIZE + mf->get_data_size());
        for(int i = 0; i < SCORD_SHADOW_SIZE; ++i)
            mf->set_data_used((addr & 127) + i);
    }
}

scord_execdata_t extract_packet_metadata(mem_fetch *mf)
{
    if(SCORD_PERF)
    {
        unsigned char *md = mf->get_scord_metadata();
        // (blkid(7) + warpid (5) + syncid (8) + atomicid (16)) / 8
        scord_execdata_t xd;
        xd.ctaid  = md[0] & 127;
        xd.warpid = md[1] & 31;
        xd.vol    = (md[1] & 64);
        xd.syncid = md[2];
        unsigned long val = md[3];
        val     <<= BLOOM_BIN_SIZE;
        val      += md[4];
        xd.lockset.setVal(val);
        xd.devfenceid = execdata[get_execid(xd.ctaid, xd.warpid)].devfenceid;
        xd.blkfenceid = execdata[get_execid(xd.ctaid, xd.warpid)].blkfenceid;
        xd.atom       = mf->isatomic();
        xd.scope = (mf->get_inst().get_atomic_scope() == 408 ? 1 : 0);
        xd.inst       = mf->get_inst().pc;
        return xd;
    }
}

EXPORT
void scord_create_packet_metadata(scord_execdata_t xd, mem_fetch *mf)
{
    if(SCORD_PERF)
    {
        unsigned char *md = (unsigned char*)mf->get_scord_metadata();
        // (blkid(7) + warpid (5) + syncid (8) + atomicid (16)) / 8
        md[0] = (xd.ctaid & 127);
        md[1] = (xd.warpid & 31);
        md[1] += (xd.vol ? 64 : 0);
        md[2] = (xd.syncid & 255);
        unsigned long val = xd.lockset.getVal();
        md[3] = ((val >> 8) & 255);
        md[4] = (val & 255);
        printf_scord("\tCTA=%d, Wid=%d, Sync=%d, atomicid=0x%x, val=0x%x\n", md[0], (int)md[1], md[2], ((int)md[3] << 8) + (int)md[4], val);
        mf->has_scord_metadata() = true;
    }
}

void scord_get_execdata(warp_inst_t *inst)
{
    if(SCORD_PERF)
    {
        scord_execdata_t &md = inst->m_scord_md;
        for(int i = 0; i < inst->warp_size(); ++i)
        {
            if(inst->active(i))
            {
                md.ctaid   = CTAID(inst->m_ldst_data[i].thread);
                md.warpid  = WARPID(inst->m_ldst_data[i].thread);
                unsigned xid    = EXECID(inst->m_ldst_data[i].thread);
                md.syncid  = execdata[xid].syncid;
                md.lockset = execdata[xid].lockset;
                md.vol = (inst->cache_op == CACHE_VOLATILE);
                return;
            }
        }
    }
}
