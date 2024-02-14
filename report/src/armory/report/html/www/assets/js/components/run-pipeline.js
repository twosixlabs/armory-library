import { computed } from 'vue';

const DownArrow = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-6 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M16 19 12 23m0 0-4-4M12 23V0" />
        </svg>
    `,
};

const DownLine = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-6 h-6">
            <path stroke-linecap="butt" stroke-linejoin="butt" d="M12 24V0" />
        </svg>
    `,
};

const TopLeftCorner = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 80 24" stroke-width="2" stroke="currentColor" class="w-20 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M40 24V1H80" />
        </svg>
    `,
};

const TopGap = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 32 2" stroke-width="2" stroke="currentColor" class="w-8 h-0.5">
            <path stroke-linecap="round" stroke-linejoin="round" d="M0 1H32" />
        </svg>
    `,
};

const TopMiddle = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 80 24" stroke-width="2" stroke="currentColor" class="w-20 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M0 1H80M40 1V24" />
        </svg>
    `,
};

const TopRightCorner = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 80 24" stroke-width="2" stroke="currentColor" class="w-20 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M40 24V1H0" />
        </svg>
    `,
};

const Split = {
    props: {
        num: Number,
    },
    components: {
        TopGap,
        TopMiddle,
        TopLeftCorner,
        TopRightCorner,
    },
    template: `
        <div class="flex">
            <TopLeftCorner></TopLeftCorner>
            <template v-for="i in num-2">
                <TopGap></TopGap>
                <TopMiddle></TopMiddle>
                <TopGap></TopGap>
            </template>
            <TopRightCorner></TopRightCorner>
        </div>
        <div class="flex gap-8">
            <slot></slot>
        </div>
    `,
};

const Chain = {
    props: {
        name: String,
    },
    template: `
        <div class="flex flex-col items-center">
            {{ name }}
            <slot></slot>
        </div>
    `,
};

const Box = {
    props: {
        input: {
            type: Boolean,
            default: false,
        },
        type: String,
        output: {
            type: Boolean,
            default: false,
        },
    },
    components: {
        DownArrow,
        DownLine,
    },
    setup(props) {
        const classes = computed(() => ({
            'bg-violet-300': props.type == "dataset",
            'border-violet-400': props.type == "dataset",
            'bg-red-300': props.type == "perturbation",
            'border-red-400': props.type == "perturbation",
            'bg-blue-300': props.type == "model",
            'border-blue-400': props.type == "model",
        }));
        return { classes };
    },
    template: `
        <div class="flex flex-col items-center w-20">
            <DownArrow v-if="input"></DownArrow>
            <div :class="classes" class="border-2 flex flex-col items-center justify-center h-20 p-2 rounded-lg shadow w-20">
                <slot></slot>
            </div>
            <DownLine v-if="output"></DownLine>
        </div>
    `,
};

export default {
    props: {
        runs: Object,
    },
    components: {
        Box,
        Chain,
        Split,
    },
    template: `
        <div class="flex flex-row justify-center mt-2">
            <div class="flex flex-col items-center">
                <Box output type="dataset">Dataset</Box>
                <Split :num="3">
                    <Chain name="benign">
                        <Box input type="model">Model</Box>
                    </Chain>
                    <Chain name="attacked">
                        <Box input type="perturbation">Attack</Box>
                        <Box input type="model">Model</Box>
                    </Chain>
                    <Chain name="defended">
                        <Box input type="perturbation">Attack</Box>
                        <Box input type="perturbation">Defense</Box>
                        <Box input type="model">Model</Box>
                    </Chain>
                </Split>
            </div>
            <div class="border-l-2 mx-4"></div>
            <div class="flex flex-col">
                <div class="flex items-center justify-center">
                    <select class="select select-bordered select-sm w-40">
                        <option disabled>First chain?</option>
                        <option selected>benign</option>
                        <option>attacked</option>
                        <option>defended</option>
                    </select>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_5_benign_no_defense.png" />
                        Label: 38
                    </div>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_5_benign_no_defense.png" />
                        Label: 38
                    </div>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_5_benign_no_defense.png" />
                        Label: 38
                    </div>
                </div>
                <div class="border-t-2 my-2"></div>
                <div class="flex items-center justify-center">
                    <select class="select select-bordered select-sm w-40">
                        <option disabled>First chain?</option>
                        <option>benign</option>
                        <option selected>attacked</option>
                        <option>defended</option>
                    </select>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_5_benign_no_defense.png" />
                        Label: 7
                    </div>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_5_benign_no_defense.png" />
                        Label: 16
                    </div>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_5_benign_no_defense.png" />
                        Label: 38
                    </div>
                </div>
            </div>
        </div>
    `,
};
