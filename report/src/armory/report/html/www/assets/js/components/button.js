export default {
    props: {
        disabled: {
            type: Boolean,
            default: false,
        },
    },
    emits: ["click"],
    computed: {
        classes() {
            return {
                'bg-twosix-grey': !this.disabled,
                'bg-zinc-300': this.disabled,
            };
        },
    },
    template: `
        <button
            :class="classes"
            class="active:bg-slate-400 border-zinc-800 flex gap-1 items-center justify-center p-2 rounded text-sm text-white uppercase"
            :disabled="disabled"
            @click="$emit('click')"
        >
            <slot></slot>
        </button>
    `,
};