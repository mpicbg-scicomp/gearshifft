#!/bin/bash

# Original script from https://github.com/gregvw/amd_sdk/
# further modifications from: https://github.com/boostorg/compute/blob/master/.travis.yml
# and: https://github.com/ddemidov/vexcl/

if [ ! -e ${AMDAPPSDKROOT}/bin/x86_64/clinfo ]; then

    # Location from which get nonce and file name from
    URL="http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/"
    URLDOWN="http://developer.amd.com/amd-license-agreement-appsdk/"

    #AMD APP SDK v3.0:
    if [[ $1 == "300" ]]; then
        echo "AMD APP SDK v3.0"
        FORM=`wget -qO - $URL | sed -n '/download-2/,/64-bit/p'`
    else
        #AMD APP SDK v2.9.1:
        echo "AMD APP SDK v2.9.1"
        FORM=`wget -qO - $URL | sed -n '/download-5/,/64-bit/p'`
    fi

    NONCE1_STRING='name="amd_developer_central_downloads_page_nonce"'
    FILE_STRING='name="f"'
    POSTID_STRING='name="post_id"'
    NONCE2_STRING='name="amd_developer_central_nonce"'

    # Get nonce from form
    NONCE1=`echo $FORM | awk -F ${NONCE1_STRING} '{print $2}'`
    NONCE1=`echo $NONCE1 | awk -F'"' '{print $2}'`
    echo $NONCE1

    # get the postid
    POSTID=`echo $FORM | awk -F ${POSTID_STRING} '{print $2}'`
    POSTID=`echo $POSTID | awk -F'"' '{print $2}'`
    echo $POSTID

    # get file name
    FILE=`echo $FORM | awk -F ${FILE_STRING} '{print $2}'`
    FILE=`echo $FILE | awk -F'"' '{print $2}'`
    echo $FILE

    FORM=`wget -qO - $URLDOWN --post-data "amd_developer_central_downloads_page_nonce=${NONCE1}&f=${FILE}&post_id=${POSTID}"`

    NONCE2=`echo $FORM | awk -F ${NONCE2_STRING} '{print $2}'`
    NONCE2=`echo $NONCE2 | awk -F'"' '{print $2}'`
    echo $NONCE2

    wget --content-disposition --trust-server-names $URLDOWN --post-data "amd_developer_central_nonce=${NONCE2}&f=${FILE}" -nc -O AMD-SDK.tar.bz2;

    if [ $? != 0 ]; then
        exit 1;
    fi

    # Unpack and install
    tar -xjf AMD-SDK.tar.bz2 || exit 1

    if [[ ${AMDAPPSDK_VERSION} == "300" ]]; then
        cp ${AMDAPPSDKROOT}/lib/x86_64/libamdocl12cl64.so ${AMDAPPSDKROOT}/lib/x86_64/sdk/libamdocl12cl64.so
    fi

    sh AMD-APP-SDK*.sh --tar -xf -C ${AMDAPPSDKROOT};

    if [ $? != 0 ]; then
        exit 1;
    fi
    chmod +x ${AMDAPPSDKROOT}/bin/x86_64/clinfo;
fi
